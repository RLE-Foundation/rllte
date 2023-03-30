import numpy as np
import torch

from hsuanwu.common.typing import *


class VanillaReplayStorage:
    """Vanilla replay storage for off-policy algorithms.

    Args:
        device: Device (cpu, cuda, ...) on which the code should be run.
        obs_shape: The data shape of observations.
        action_shape: The data shape of actions.
        action_type: The type of actions, 'cont' or 'dis'.
        buffer_size: Max number of element in the buffer.
        batch_size: Batch size of samples.

    Returns:
        Vanilla replay storage.
    """

    def __init__(
        self,
        device: torch.device,
        obs_shape: Tuple,
        action_shape: Tuple,
        action_type: str,
        buffer_size: int = 1e6,
        batch_size: int = 1024,
    ):
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._device = torch.device(device)
        self._buffer_size = buffer_size
        self._batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obs = np.empty((buffer_size, *obs_shape), dtype=obs_dtype)

        if action_type == "dis":
            self.actions = self.actions = np.empty((buffer_size, 1), dtype=np.float32)
        if action_type == "cont":
            self.actions = np.empty((buffer_size, action_shape[0]), dtype=np.float32)

        self.rewards = np.empty((buffer_size, 1), dtype=np.float32)
        self.dones = np.empty((buffer_size, 1), dtype=np.float32)

        self._global_step = 0
        self._full = False

    def __len__(self):
        return self._buffer_size if self._full else self._global_step

    def add(
        self, obs: Any, action: Any, reward: Any, done: Any, info: Any, next_obs: Any
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            obs: Observations.
            actions: Actions.
            rewards: Rewards.
            dones: Dones.
            info: Infos.
            next_obs: Next observations.

        Returns:
            None.
        """
        np.copyto(self.obs[self._global_step], obs)
        np.copyto(self.actions[self._global_step], action)
        np.copyto(self.rewards[self._global_step], reward)
        np.copyto(self.obs[(self._global_step + 1) % self._buffer_size], next_obs)
        np.copyto(self.dones[self._global_step], done)

        self._global_step = (self._global_step + 1) % self._buffer_size
        self._full = self._full or self._global_step == 0

    def sample(self) -> Batch:
        """Sample transitions from the storage."""
        indices = np.random.randint(
            0,
            self._buffer_size if self._full else self._global_step,
            size=self._batch_size,
        )

        obs = torch.as_tensor(self.obs[indices], device=self._device).float()
        actions = torch.as_tensor(self.actions[indices], device=self._device).float()
        rewards = torch.as_tensor(self.rewards[indices], device=self._device).float()
        next_obs = torch.as_tensor(
            self.obs[(indices + 1) % self._buffer_size], device=self._device
        ).float()
        dones = torch.as_tensor(self.dones[indices], device=self._device).float()

        return obs, actions, rewards, dones, next_obs
