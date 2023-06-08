from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.base_storage import BaseStorage


class VanillaReplayStorage(BaseStorage):
    """Vanilla replay storage for off-policy algorithms.

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        storage_size (int): Max number of element in the buffer.
        batch_size (int): Batch size of samples.

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
    ):
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

        self.global_step = 0
        self.full = False

    def __len__(self):
        return self.storage_size if self.full else self.global_step

    def add(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        terminated: Any,
        info: Any,
        next_obs: Any,
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            obs (Any): Observations.
            action (Any): Actions.
            reward (Any): Rewards.
            terminated (Any): Terminateds.
            info (Any): Infos.
            next_obs (Any): Next observations.

        Returns:
            None.
        """
        np.copyto(self.obs[self.global_step], obs)
        np.copyto(self.actions[self.global_step], action)
        np.copyto(self.rewards[self.global_step], reward)
        np.copyto(self.obs[(self.global_step + 1) % self.storage_size], next_obs)
        np.copyto(self.terminateds[self.global_step], terminated)

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
        weights = th.ones_like(terminateds, device=self.device)

        return indices, obs, actions, rewards, terminateds, next_obs, weights

    def update(self, *args) -> None:
        """Update the storage if necessary."""
        raise NotImplementedError
