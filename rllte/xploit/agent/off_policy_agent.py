# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common import utils
from rllte.common.base_agent import BaseAgent
from rllte.common.policies import OffPolicyDeterministicActorDoubleCritic, OffPolicyStochasticActorDoubleCritic
from rllte.xploit.encoder import IdentityEncoder, TassaCnnEncoder
from rllte.xploit.storage import NStepReplayStorage, VanillaReplayStorage
from rllte.xplore.augmentation import Identity, RandomShift
from rllte.xplore.distribution import SquashedNormal, TruncatedNormalNoise


class OffPolicyAgent(BaseAgent):
    """Trainer for off-policy algorithms.

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
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
            self.policy = OffPolicyDeterministicActorDoubleCritic(
                action_dim=self.action_dim, feature_dim=self.feature_dim, hidden_dim=hidden_dim
            )
            self.storage = NStepReplayStorage(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                batch_size=batch_size,
            )
            self.dist = TruncatedNormalNoise(low=self.action_range[0], high=self.action_range[1])
            # for `DDPG` without augmentation
            if len(self.obs_shape) == 1:
                self.aug = Identity().to(self.device)
            else:
                self.aug = RandomShift(pad=4).to(self.device)

        if kwargs["agent_name"] == "SAC":
            self.policy = OffPolicyStochasticActorDoubleCritic(
                action_dim=self.action_dim, feature_dim=self.feature_dim, hidden_dim=hidden_dim
            )
            self.storage = VanillaReplayStorage(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                batch_size=batch_size,
            )

            # build distribution
            self.dist = SquashedNormal

    def update(self) -> Dict[str, float]:
        """Update the agent. Implemented by individual algorithms."""
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

        # training loop
        while self.global_step <= num_train_steps:
            # try to eval
            if (self.global_step % self.eval_every_steps) == 0 and (self.eval_env is not None):
                eval_metrics = self.eval()
                self.logger.eval(msg=eval_metrics)

            # sample actions
            with th.no_grad(), utils.eval_mode(self):
                # Initial exploration
                if self.global_step <= self.num_init_steps:
                    action = th.rand(size=(self.num_envs, self.action_dim), device=self.device).uniform_(-1.0, 1.0)
                else:
                    action = self.policy(obs, training=True, step=self.global_step)

            # observe reward and next obs
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward[0].cpu().numpy()
            episode_step += 1
            self.global_step += 1

            # TODO: add parallel env support
            # save transition
            reward = th.zeros_like(reward) if self.pretraining else reward  # pre-training mode
            self.storage.add(
                obs=obs[0].cpu().numpy(),
                action=action[0].cpu().numpy(),
                reward=reward[0].cpu().numpy(),
                terminated=terminated[0].cpu().numpy(),
                truncated=truncated[0].cpu().numpy(),
                info=info,
                next_obs=next_obs[0].cpu().numpy(),
            )

            # update agent
            if self.global_step >= self.num_init_steps:
                metrics = self.update()
                # try to update storage
                self.storage.update(metrics)

            # terminated or truncated
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

                # As the vector environments autoreset for a terminating and truncating sub-environments,
                # the returned observation and info is not the final step's observation or info which
                # is instead stored in info as `final_observation` and `final_info`. Therefore,
                # we don't need to reset the env here.
                self.global_episode += 1
                episode_step, episode_reward = 0, 0

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
        # reset the env
        step, episode, total_reward = 0, 0, 0
        obs, info = self.eval_env.reset(seed=self.seed)

        # eval loop
        while episode <= self.num_eval_episodes:
            # sample actions
            with th.no_grad(), utils.eval_mode(self):
                action = self.policy(obs, training=False, step=self.global_step)

            # observe reward and next obs
            next_obs, reward, terminated, truncated, info = self.eval_env.step(action)
            total_reward += reward[0].cpu().numpy()
            step += 1

            # terminated or truncated
            if terminated or truncated:
                # As the vector environments autoreset for a terminating and truncating sub-environments,
                # the returned observation and info is not the final step's observation or info which
                # is instead stored in info as `final_observation` and `final_info`. Therefore,
                # we don't need to reset the env here.
                episode += 1

            obs = next_obs

        return {
            "step": self.global_step,
            "episode": self.global_episode,
            "episode_length": step / episode,
            "episode_reward": total_reward / episode,
            "total_time": self.timer.total_time(),
        }
