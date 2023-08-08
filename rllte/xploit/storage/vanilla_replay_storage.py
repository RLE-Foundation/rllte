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


from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.base_storage import BaseStorage, VanillaReplayBatch

class VanillaReplayStorage(BaseStorage):
    """Vanilla replay storage for off-policy algorithms.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        device (str): Device to store the data.
        storage_size (int): Storage size.
        num_envs (int): The number of parallel environments.
        batch_size (int): Batch size.

    Returns:
        Vanilla replay storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        storage_size: int = 1000000,
        num_envs: int = 1,
        batch_size: int = 1024,
    ) -> None:
        super().__init__(observation_space, action_space, device)
        # split the storage size for each environment
        self.storage_size = max(storage_size // num_envs, 1)
        self.num_envs = num_envs
        self.batch_size = batch_size
        
        # data containers
        ###########################################################################################################
        # initialize this container later, for `DictReplayStorage`
        if not isinstance(self.observation_space, gym.spaces.Dict):
            self.observations = np.empty((self.storage_size, num_envs, *self.obs_shape), dtype=observation_space.dtype)
        if self.action_type == "Discrete":
            self.actions = np.empty((self.storage_size, num_envs), dtype=np.int64)
        else:
            self.actions = np.empty((self.storage_size, num_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.empty((self.storage_size, num_envs), dtype=np.float32)
        self.terminateds = np.empty((self.storage_size, num_envs), dtype=np.float32)
        self.truncateds = np.empty((self.storage_size, num_envs), dtype=np.float32)
        ###########################################################################################################

        # counter
        self.step = 0
        self.full = False

    def __len__(self) -> int:
        """Return the number of transitions in storage."""
        return self.storage_size if self.full else self.step

    def add(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        infos: Dict[str, Any],
        next_observations: th.Tensor,
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            observations (th.Tensor): Observations.
            actions (th.Tensor): Actions.
            rewards (th.Tensor): Rewards.
            terminateds (th.Tensor): Termination flag.
            truncateds (th.Tensor): Truncation flag.
            infos (Dict[str, Any]): Additional information.
            next_observations (th.Tensor): Next observations.

        Returns:
            None.
        """
        np.copyto(self.observations[self.step], observations.cpu().numpy())
        np.copyto(self.actions[self.step], actions.cpu().numpy())
        np.copyto(self.rewards[self.step], rewards.cpu().numpy())
        np.copyto(self.observations[(self.step + 1) % self.storage_size], next_observations.cpu().numpy())
        np.copyto(self.terminateds[self.step], terminateds.cpu().numpy())
        np.copyto(self.truncateds[self.step], truncateds.cpu().numpy())

        self.step = (self.step + 1) % self.storage_size
        self.full = self.full or self.step == 0

    def sample(self, step: int) -> VanillaReplayBatch:
        """Sample from the storage.

        Args:
            step (int): Global training step.

        Returns:
            Batched samples.
        """
        # get batch and env indices
        if self.full:
            batch_indices = (np.random.randint(1, self.storage_size, size=self.batch_size) + self.step) % self.storage_size
        else:
            batch_indices = np.random.randint(0, self.step, size=self.batch_size)
        env_indices = np.random.randint(0, self.num_envs, size=(self.batch_size, ))

        # get batch data
        obs = self.observations[batch_indices, env_indices, :]
        actions = self.actions[batch_indices, env_indices]
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, 1)
        terminateds = self.terminateds[batch_indices, env_indices].reshape(-1, 1)
        truncateds = self.truncateds[batch_indices, env_indices].reshape(-1, 1)
        next_obs = self.observations[(batch_indices + 1) % self.storage_size, env_indices, :]

        return VanillaReplayBatch(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            rewards=self.to_torch(rewards),
            terminateds=self.to_torch(terminateds),
            truncateds=self.to_torch(truncateds),
            next_observations=self.to_torch(next_obs)
        )

    def update(self, *args) -> None:
        """Update the storage if necessary."""
        return None
