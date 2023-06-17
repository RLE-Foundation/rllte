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

from collections import deque
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common import utils
from rllte.common.base_agent import BaseAgent
from rllte.common.policies import OnPolicyDecoupledActorCritic, OnPolicySharedActorCritic
from rllte.xploit.encoder import IdentityEncoder, PathakCnnEncoder
from rllte.xploit.storage import VanillaRolloutStorage as Storage
from rllte.xplore.distribution import Bernoulli, Categorical, DiagonalGaussian


class OnPolicyAgent(BaseAgent):
    """Trainer for on-policy algorithms.

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.
        num_steps (int): The sample length of per rollout.
        eval_every_episodes (int): Evaluation interval.
        shared_encoder (bool): `True` for using a shared encoder, `False` for using two separate encoders.
        **kwargs: Arbitrary arguments such as `batch_size` and `hidden_dim`.

    Returns:
        On-policy agent instance.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_steps: int = 128,
        eval_every_episodes: int = 10,
        shared_encoder: bool = True,
        **kwargs,
    ) -> None:
        feature_dim = kwargs.pop("feature_dim", 512)
        hidden_dim = kwargs.pop("feature_dim", 256)
        batch_size = kwargs.pop("batch_size", 256)
        super().__init__(
            env=env, eval_env=eval_env, tag=tag, seed=seed, device=device, pretraining=pretraining, feature_dim=feature_dim
        )
        self.num_steps = num_steps
        self.eval_every_episodes = eval_every_episodes

        # build encoder
        if len(self.obs_shape) == 3:
            self.encoder = PathakCnnEncoder(observation_space=env.observation_space, feature_dim=self.feature_dim)
        elif len(self.obs_shape) == 1:
            self.feature_dim = self.obs_shape[0]
            self.encoder = IdentityEncoder(observation_space=env.observation_space, feature_dim=self.feature_dim)

        # build storage
        self.storage = Storage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            batch_size=batch_size,
        )

        # build distribution
        if self.action_type == "Discrete":
            self.dist = Categorical
        elif self.action_type == "Box":
            self.dist = DiagonalGaussian
        elif self.action_type == "MultiBinary":
            self.dist = Bernoulli
        else:
            raise NotImplementedError("Unsupported action type!")

        # create policy
        if shared_encoder:
            self.policy = OnPolicySharedActorCritic(
                obs_shape=self.obs_shape,
                action_dim=self.action_dim,
                action_type=self.action_type,
                feature_dim=self.feature_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.policy = OnPolicyDecoupledActorCritic(
                obs_shape=self.obs_shape,
                action_dim=self.action_dim,
                action_type=self.action_type,
                feature_dim=self.feature_dim,
                hidden_dim=hidden_dim,
            )

    def update(self) -> Dict[str, float]:
        """Update the agent. Implemented by individual algorithms."""
        raise NotImplementedError

    def freeze(self) -> None:
        """Freeze the structure of the agent. Implemented by individual algorithms."""
        raise NotImplementedError

    def mode(self, training: bool = True) -> None:
        """Set the training mode.

        Args:
            training (bool): True (training) or False (evaluation).

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
        episode_rewards = deque(maxlen=100)
        episode_steps = deque(maxlen=100)
        obs, info = self.env.reset(seed=self.seed)
        # Number of updates
        num_updates = num_train_steps // self.num_envs // self.num_steps

        for update in range(num_updates):
            # try to eval
            if (update % self.eval_every_episodes) == 0 and (self.eval_env is not None):
                eval_metrics = self.eval()
                self.logger.eval(msg=eval_metrics)

            for _step in range(self.num_steps):
                # sample actions
                with th.no_grad(), utils.eval_mode(self):
                    actions, values, log_probs = self.policy.get_action_and_value(obs, training=True)

                (
                    next_obs,
                    rewards,
                    terminateds,
                    truncateds,
                    infos,
                ) = self.env.step(actions.clamp(*self.action_range))

                if "episode" in infos:
                    indices = np.nonzero(infos["episode"]["l"])
                    episode_rewards.extend(infos["episode"]["r"][indices].tolist())
                    episode_steps.extend(infos["episode"]["l"][indices].tolist())

                # add transitions
                self.storage.add(
                    obs=obs,
                    actions=actions,
                    rewards=th.zeros_like(rewards, device=self.device) if self.pretraining else rewards,  # pre-training mode
                    terminateds=terminateds,
                    truncateds=truncateds,
                    next_obs=next_obs,
                    log_probs=log_probs,
                    values=values,
                )

                obs = next_obs

            # get the value estimation of the last step
            with th.no_grad():
                last_values = self.policy.get_value(next_obs).detach()

            # compute intrinsic rewards
            if self.irs is not None:
                intrinsic_rewards = self.irs.compute_irs(
                    samples={
                        "obs": self.storage.obs[:-1],
                        "actions": self.storage.actions,
                        "next_obs": self.storage.obs[1:],
                    },
                    step=self.global_episode * self.num_envs * self.num_steps,
                )
                self.storage.rewards += intrinsic_rewards.to(self.device)

            # perform return and advantage estimation
            self.storage.compute_returns_and_advantages(last_values)

            # agent update
            self.update()

            # update and reset buffer
            self.storage.update()

            self.global_episode += 1
            self.global_step += self.num_envs * self.num_steps
            episode_time, total_time = self.timer.reset()

            if len(episode_rewards) > 1:
                train_metrics = {
                    "step": self.global_step,
                    "episode": self.global_episode * self.num_envs,
                    "episode_length": np.mean(episode_steps),
                    "episode_reward": np.mean(episode_rewards),
                    "fps": self.num_steps * self.num_envs / episode_time,
                    "total_time": total_time,
                }
                self.logger.train(msg=train_metrics)

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
        obs, info = self.eval_env.reset(seed=self.seed)
        episode_rewards = list()
        episode_steps = list()

        while len(episode_rewards) < self.num_eval_episodes:
            with th.no_grad(), utils.eval_mode(self):
                actions = self.policy.get_action_and_value(obs, training=False)
            obs, rewards, terminateds, truncateds, infos = self.eval_env.step(actions.clamp(*self.action_range))

            if "episode" in infos:
                indices = np.nonzero(infos["episode"]["l"])
                episode_rewards.extend(infos["episode"]["r"][indices].tolist())
                episode_steps.extend(infos["episode"]["l"][indices].tolist())

        return {
            "step": self.global_step,
            "episode": self.global_episode,
            "episode_length": np.mean(episode_steps),
            "episode_reward": np.mean(episode_rewards),
            "total_time": self.timer.total_time(),
        }
