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

from rllte.common.base_storage import BaseStorage


class VanillaReplayStorage(BaseStorage):
    """Vanilla replay storage for off-policy algorithms.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        device (str): Device to store the data.
        storage_size (int): Storage size.
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
        batch_size: int = 1024,
    ) -> None:
        super().__init__(observation_space, action_space, device)
        self.storage_size = storage_size
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(self.obs_shape) == 1 else np.uint8

        self.obs = np.empty((storage_size, *self.obs_shape), dtype=obs_dtype)
        if self.action_type == "Discrete":
            self.actions = np.empty((storage_size, 1), dtype=np.float32)
        if self.action_type == "Box":
            self.actions = np.empty((storage_size, self.action_shape[0]), dtype=np.float32)
        self.rewards = np.empty((storage_size, 1), dtype=np.float32)
        self.terminateds = np.empty((storage_size, 1), dtype=np.float32)
        self.truncateds = np.empty((storage_size, 1), dtype=np.float32)

        self.global_step = 0
        self.full = False

    def __len__(self) -> int:
        """Return the number of transitions in storage."""
        return self.storage_size if self.full else self.global_step

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: Dict[str, Any],
        next_obs: np.ndarray,
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            obs (np.ndarray): Observation.
            action (np.ndarray): Action.
            reward (np.ndarray): Reward.
            terminated (np.ndarray): Termination flag.
            truncated (np.ndarray): Truncation flag.
            info (Dict[str, Any]): Additional information.
            next_obs (np.ndarray): Next observation.

        Returns:
            None.
        """
        # TODO: add parallel env support
        np.copyto(self.obs[self.global_step], obs[0].cpu().numpy())
        np.copyto(self.actions[self.global_step], action[0].cpu().numpy())
        np.copyto(self.rewards[self.global_step], reward[0].cpu().numpy())
        np.copyto(self.obs[(self.global_step + 1) % self.storage_size], next_obs[0].cpu().numpy())
        np.copyto(self.terminateds[self.global_step], terminated[0].cpu().numpy())
        np.copyto(self.truncateds[self.global_step], truncated[0].cpu().numpy())

        self.global_step = (self.global_step + 1) % self.storage_size
        self.full = self.full or self.global_step == 0

    def sample(self, step: int) -> Tuple[th.Tensor, ...]:
        """Sample from the storage.

        Args:
            step (int): Global training step.

        Returns:
            Batched samples.
        """
        indices = np.random.randint(
            0,
            self.storage_size if self.full else self.global_step,
            size=self.batch_size,
        )

        obs = th.as_tensor(self.obs[indices], device=self.device).float()
        actions = th.as_tensor(self.actions[indices], device=self.device).float()
        rewards = th.as_tensor(self.rewards[indices], device=self.device).float()
        next_obs = th.as_tensor(self.obs[(indices + 1) % self.storage_size], device=self.device).float()
        terminateds = th.as_tensor(self.terminateds[indices], device=self.device).float()
        truncateds = th.as_tensor(self.truncateds[indices], device=self.device).float()
        weights = th.ones_like(terminateds, device=self.device)

        return indices, obs, actions, rewards, terminateds, truncateds, next_obs, weights

    def update(self, *args) -> None:
        """Update the storage if necessary."""
        return None
