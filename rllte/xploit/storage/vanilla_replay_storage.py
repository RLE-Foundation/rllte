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
from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.base_storage import BaseStorage

Batch = namedtuple(typename="Batch", field_names=[
    "obs", 
    "actions",
    "rewards",
    "terminateds",
    "truncateds",
    "next_obs"
])

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
        self.obs = np.empty((self.storage_size, num_envs, *self.obs_shape), dtype=observation_space.dtype)
        if self.action_type == "Discrete":
            self.actions = np.empty((self.storage_size, num_envs), dtype=np.int64)
        else:
            self.actions = np.empty((self.storage_size, num_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.empty((self.storage_size, num_envs), dtype=np.float32)
        self.terminateds = np.empty((self.storage_size, num_envs), dtype=np.float32)
        self.truncateds = np.empty((self.storage_size, num_envs), dtype=np.float32)

        # counter
        self.global_step = 0
        self.full = False

    def __len__(self) -> int:
        """Return the number of transitions in storage."""
        return self.storage_size if self.full else self.global_step

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminateds: np.ndarray,
        truncateds: np.ndarray,
        info: Dict[str, Any],
        next_obs: np.ndarray,
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            obs (np.ndarray): Observations.
            actions (np.ndarray): Actions.
            rewards (np.ndarray): Rewards.
            terminateds (np.ndarray): Termination flag.
            truncateds (np.ndarray): Truncation flag.
            info (Dict[str, Any]): Additional information.
            next_obs (np.ndarray): Next observations.

        Returns:
            None.
        """
        np.copyto(self.obs[self.global_step], obs.cpu().numpy())
        np.copyto(self.actions[self.global_step], actions.cpu().numpy())
        np.copyto(self.rewards[self.global_step], rewards.cpu().numpy())
        np.copyto(self.obs[(self.global_step + 1) % self.storage_size], next_obs.cpu().numpy())
        np.copyto(self.terminateds[self.global_step], terminateds.cpu().numpy())
        np.copyto(self.truncateds[self.global_step], truncateds.cpu().numpy())

        self.global_step = (self.global_step + 1) % self.storage_size
        self.full = self.full or self.global_step == 0

    def sample(self, step: int) -> Batch:
        """Sample from the storage.

        Args:
            step (int): Global training step.

        Returns:
            Batched samples.
        """
        # get indices
        batch_indices = np.random.randint(
            0,
            self.storage_size if self.full else self.global_step,
            size=self.batch_size,
        )
        env_indices = np.random.randint(0, self.num_envs, size=(self.batch_size, ))

        # get batch data
        obs = th.as_tensor(self.obs[batch_indices, env_indices, :], device=self.device).float()
        actions = th.as_tensor(self.actions[batch_indices, env_indices, :], device=self.device).float()
        rewards = th.as_tensor(self.rewards[batch_indices, env_indices].reshape(-1, 1), device=self.device).float()
        next_obs = th.as_tensor(self.obs[(batch_indices + 1) % self.storage_size, env_indices, :], device=self.device).float()
        terminateds = th.as_tensor(self.terminateds[batch_indices, env_indices].reshape(-1, 1), device=self.device).float()
        truncateds = th.as_tensor(self.truncateds[batch_indices, env_indices].reshape(-1, 1), device=self.device).float()

        return Batch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            next_obs=next_obs
        )

    def update(self, *args) -> None:
        """Update the storage if necessary."""
        return None
