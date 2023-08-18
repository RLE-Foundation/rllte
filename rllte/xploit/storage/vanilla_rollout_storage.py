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


from typing import Dict, Generator

import gymnasium as gym
import numpy as np
import torch as th
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from rllte.common.prototype import BaseStorage
from rllte.common.type_alias import VanillaRolloutBatch


class VanillaRolloutStorage(BaseStorage):
    """Vanilla rollout storage for on-policy algorithms.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        storage_size (int): The size of the storage.
        batch_size (int): Batch size of samples.
        num_envs (int): The number of parallel environments.
        discount (float): The discount factor.
        gae_lambda (float): Weighting coefficient for generalized advantage estimation (GAE).

    Returns:
        Vanilla rollout storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        storage_size: int = 256,
        batch_size: int = 64,
        num_envs: int = 8,
        discount: float = 0.999,
        gae_lambda: float = 0.95,
    ) -> None:
        super().__init__(observation_space, action_space, device, storage_size, batch_size, num_envs)
        self.storage_size = storage_size
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.reset()
    
    def reset(self) -> None:
        """Reset the storage."""
        # data containers
        self.observations = np.empty(shape=(self.storage_size + 1, self.num_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty(shape=(self.storage_size, self.num_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.empty(shape=(self.storage_size, self.num_envs), dtype=np.float32)
        self.terminateds = np.empty(shape=(self.storage_size + 1, self.num_envs), dtype=np.float32)
        self.truncateds = np.empty(shape=(self.storage_size + 1, self.num_envs), dtype=np.float32)
        # first next_terminated
        self.terminateds[0].fill(0.0)
        self.truncateds[0].fill(0.0)
        # extra part
        self.log_probs = np.empty(shape=(self.storage_size, self.num_envs), dtype=np.float32)
        self.values = np.empty(shape=(self.storage_size, self.num_envs), dtype=np.float32)
        self.returns = np.empty(shape=(self.storage_size, self.num_envs), dtype=np.float32)
        self.advantages = np.empty(shape=(self.storage_size, self.num_envs), dtype=np.float32)
        super().reset()

    def add(self,
            observations: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminateds: np.ndarray,
            truncateds: np.ndarray,
            infos: Dict,
            next_observations: np.ndarray,
            log_probs: th.Tensor,
            values: th.Tensor
            ) -> None:
        """Add sampled transitions into storage.

        Args:
            observations (np.ndarray): Observations.
            actions (np.ndarray): Actions.
            rewards (np.ndarray): Rewards.
            terminateds (np.ndarray): Termination signals.
            truncateds (np.ndarray): Truncation signals.
            infos (Dict): Extra information.
            next_observations (np.ndarray): Next observations.
            log_probs (th.Tensor): Log of the probability evaluated at `actions`.
            values (th.Tensor): Estimated values.

        Returns:
            None.
        """
        np.copyto(self.observations[self.step], observations)
        np.copyto(self.actions[self.step], actions.reshape((self.num_envs, self.action_dim)))
        np.copyto(self.rewards[self.step], rewards)
        np.copyto(self.terminateds[self.step + 1], terminateds)
        np.copyto(self.truncateds[self.step + 1], truncateds)
        np.copyto(self.observations[self.step + 1], next_observations)
        np.copyto(self.log_probs[self.step], log_probs.cpu().numpy())
        np.copyto(self.values[self.step], values.cpu().numpy().flatten())

        self.step = (self.step + 1) % self.storage_size

    def update(self) -> None:
        """Reset the terminal state of each env."""
        np.copyto(self.terminateds[0], self.terminateds[-1])
        np.copyto(self.truncateds[0], self.truncateds[-1])

    def compute_returns_and_advantages(self, last_values: th.Tensor) -> None:
        """Perform generalized advantage estimation (GAE).

        Args:
            last_values (th.Tensor): Estimated values of the last step.

        Returns:
            None.
        """
        gae = 0
        for step in reversed(range(self.storage_size)):
            if step == self.storage_size - 1:
                next_values = last_values.cpu().numpy()[:, 0]
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.terminateds[step + 1]
            delta = self.rewards[step] + self.discount * next_values * next_non_terminal - self.values[step]
            gae = delta + self.discount * self.gae_lambda * next_non_terminal * gae
            # time limit
            gae = gae * (1.0 - self.truncateds[step + 1])
            self.advantages[step] = gae

        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)

    def sample(self) -> Generator:
        """Sample data from storage."""
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_envs * self.storage_size)), self.batch_size, drop_last=True)

        for indices in sampler:
            batch_obs = self.observations[:-1].reshape((-1, *self.obs_shape))[indices]
            batch_actions = self.actions.reshape((-1, *self.action_shape))[indices]
            batch_values = self.values.flatten()[indices]
            batch_returns = self.returns.flatten()[indices]
            batch_terminateds = self.terminateds[:-1].flatten()[indices]
            batch_truncateds = self.truncateds[:-1].flatten()[indices]
            batch_old_log_probs = self.log_probs.flatten()[indices]
            adv_targ = self.advantages.flatten()[indices]

            yield VanillaRolloutBatch(
                observations=self.to_torch(batch_obs),
                actions=self.to_torch(batch_actions),
                values=self.to_torch(batch_values),
                returns=self.to_torch(batch_returns),
                terminateds=self.to_torch(batch_terminateds),
                truncateds=self.to_torch(batch_truncateds),
                old_log_probs=self.to_torch(batch_old_log_probs),
                adv_targ=self.to_torch(adv_targ)
            )
