from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common import utils
from rllte.common.base_agent import BaseAgent
from rllte.common.policies import (
    NpuOffPolicyDeterministicActorDoubleCritic,
    NpuOffPolicyStochasticActorDoubleCritic,
    OffPolicyDeterministicActorDoubleCritic,
    OffPolicyStochasticActorDoubleCritic,
)
from rllte.xploit.encoder import IdentityEncoder, TassaCnnEncoder
from rllte.xploit.storage import NStepReplayStorage, VanillaReplayStorage
from rllte.xplore.augmentation import RandomShift
from rllte.xplore.distribution import SquashedNormal, TruncatedNormalNoise


class OffPolicyAgent(BaseAgent):
    """Trainer for off-policy algorithms.

    Args:
        env (Env): A Gym-like environment for training.
        eval_env (Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.
        num_init_steps (int): Number of initial exploration steps.
        eval_every_steps (int): Evaluation interval.
        **kwargs: Arbitrary arguments such as `batch_size` and `hidden_dim`.

    Returns:
        Off-policy agent instance.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_init_steps: int = 2000,
        eval_every_steps: int = 5000,
        **kwargs,
    ) -> None:
        feature_dim = kwargs.pop("feature_dim", 50)
        hidden_dim = kwargs.pop("feature_dim", 1024)
        batch_size = kwargs.pop("batch_size", 256)
        npu = kwargs.pop("npu", False)
        super().__init__(
            env=env, eval_env=eval_env, tag=tag, seed=seed, device=device, pretraining=pretraining, feature_dim=feature_dim
        )

        self.eval_every_steps = eval_every_steps
        self.num_init_steps = num_init_steps

        # build encoder
        if len(self.obs_shape) == 3:
            self.encoder = TassaCnnEncoder(observation_space=env.observation_space, feature_dim=self.feature_dim)
        elif len(self.obs_shape) == 1:
            self.feature_dim = self.obs_shape[0]
            self.encoder = IdentityEncoder(observation_space=env.observation_space, feature_dim=self.feature_dim)

        if kwargs["agent_name"] == "DrQv2":
            if npu:
                self.policy = NpuOffPolicyDeterministicActorDoubleCritic(
                    action_dim=self.action_dim, feature_dim=self.feature_dim, hidden_dim=hidden_dim
                )
            else:
                self.policy = OffPolicyDeterministicActorDoubleCritic(
                    action_dim=self.action_dim, feature_dim=self.feature_dim, hidden_dim=hidden_dim
                )
            self.storage = NStepReplayStorage(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device="cpu" if npu else device,
                batch_size=batch_size,
            )
            self.dist = TruncatedNormalNoise()
            self.aug = RandomShift(pad=4)

        if kwargs["agent_name"] == "SAC":
            if npu:
                self.policy = NpuOffPolicyStochasticActorDoubleCritic(
                    action_dim=self.action_dim, feature_dim=self.feature_dim, hidden_dim=hidden_dim
                )
            else:
                self.policy = OffPolicyStochasticActorDoubleCritic(
                    action_dim=self.action_dim, feature_dim=self.feature_dim, hidden_dim=hidden_dim
                )
            self.storage = VanillaReplayStorage(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device="cpu" if npu else device,
                batch_size=batch_size,
            )

            # build distribution
            self.dist = SquashedNormal

    def update(self) -> Dict[str, float]:
        """Update function of the agent. Implemented by individual algorithms."""
        raise NotImplementedError

    def freeze(self) -> None:
        """Freeze the structure of the agent. Implemented by individual algorithms."""
        raise NotImplementedError

    def mode(self, training: bool = True) -> None:
        """Set the training mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self.policy.train(training)

    def train(self, num_train_steps: int = 100000, init_model_path: Optional[str] = None) -> None:
        """Training function.

        Args:
            num_train_steps (int): Number of training steps.
            init_model_path (Optional[str]): Path of Iinitial model parameters.

        Returns:
            None.
        """
        # freeze the structure of the agent
        self.freeze()
        # final check
        self.check()
        # load initial model parameters
        if init_model_path is not None:
            self.logger.info(f"Loading Initial Parameters from {init_model_path}...")
            self.policy.load(init_model_path)
        # reset the env
        episode_step, episode_reward = 0, 0
        obs, info = self.env.reset(seed=self.seed)
        metrics = None

        while self.global_step <= num_train_steps:
            # try to eval
            if (self.global_step % self.eval_every_steps) == 0 and (self.eval_env is not None):
                eval_metrics = self.eval()
                self.logger.eval(msg=eval_metrics)

            # sample actions
            with th.no_grad(), utils.eval_mode(self):
                action = self.policy(obs, training=True, step=self.global_step)
                # Initial exploration
                if self.global_step < self.num_init_steps:
                    action.uniform_(-1.0, 1.0)
            next_obs, reward, terminated, truncated, info = self.env.step(action.clamp(*self.action_range))
            episode_reward += reward[0].cpu().numpy()
            episode_step += 1
            self.global_step += 1

            # save transition
            self.storage.add(
                obs[0].cpu().numpy(),
                action[0].cpu().numpy(),
                np.zeros_like(reward[0].cpu().numpy()) if self.pretraining else reward[0].cpu().numpy(),  # pre-training mode
                terminated[0].cpu().numpy(),
                info,
                next_obs[0].cpu().numpy(),
            )

            # update agent
            if self.global_step >= self.num_init_steps:
                metrics = self.update()
                # try to update storage
                self.storage.update(metrics)

            # done
            if terminated or truncated:
                episode_time, total_time = self.timer.reset()
                if metrics is not None:
                    train_metrics = {
                        "step": self.global_step,
                        "episode": self.global_episode,
                        "episode_length": episode_step,
                        "episode_reward": episode_reward,
                        "fps": episode_step / episode_time,
                        "total_time": total_time,
                    }
                    self.logger.train(msg=train_metrics)

                obs, info = self.env.reset(seed=self.seed)
                self.global_episode += 1
                episode_step, episode_reward = 0, 0
                continue

            obs = next_obs

        # save model
        self.logger.info("Training Accomplished!")
        if self.pretraining:  # pretraining
            save_dir = Path.cwd() / "pretrained"
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = Path.cwd() / "model"
            save_dir.mkdir(exist_ok=True)
        self.policy.save(path=save_dir, pretraining=self.pretraining)
        self.logger.info(f"Model saved at: {save_dir}")

        # close env
        self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()

    def eval(self) -> Dict[str, float]:
        """Evaluation function."""
        step, episode, total_reward = 0, 0, 0
        obs, info = self.eval_env.reset(seed=self.seed)

        while episode <= self.num_eval_episodes:
            with th.no_grad(), utils.eval_mode(self):
                action = self.policy(obs, training=False, step=self.global_step)

            next_obs, reward, terminated, truncated, info = self.eval_env.step(action.clamp(*self.action_range))
            total_reward += reward[0].cpu().numpy()
            step += 1

            if terminated or truncated:
                obs, info = self.eval_env.reset(seed=self.seed)
                episode += 1
                continue

            obs = next_obs

        return {
            "step": self.global_step,
            "episode": self.global_episode,
            "episode_length": step / episode,
            "episode_reward": total_reward / episode,
            "total_time": self.timer.total_time(),
        }
