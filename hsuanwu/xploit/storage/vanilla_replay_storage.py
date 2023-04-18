from typing import Any, Tuple
import numpy as np
import torch as th

class VanillaReplayStorage:
    """Vanilla replay storage for off-policy algorithms.

    Args:
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        obs_shape (Tuple): The data shape of observations.
        action_shape (Tuple): The data shape of actions.
        action_type (str): The type of actions, 'Discrete' or 'Box'.
        storage_size (int): Max number of element in the buffer.
        batch_size (int): Batch size of samples.

    Returns:
        Vanilla replay storage.
    """

    def __init__(
        self,
        device: th.device,
        obs_shape: Tuple,
        action_shape: Tuple,
        action_type: str,
        storage_size: int = 1e6,
        batch_size: int = 1024,
    ):
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._device = th.device(device)
        self._storage_size = storage_size
        self._batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obs = np.empty((storage_size, *obs_shape), dtype=obs_dtype)

        if action_type == "Discrete":
            self.actions = self.actions = np.empty((storage_size, 1), dtype=np.float32)
        if action_type == "Box":
            self.actions = np.empty((storage_size, action_shape[0]), dtype=np.float32)

        self.rewards = np.empty((storage_size, 1), dtype=np.float32)
        self.terminateds = np.empty((storage_size, 1), dtype=np.float32)

        self._global_step = 0
        self._full = False

    def __len__(self):
        return self._storage_size if self._full else self._global_step

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
        np.copyto(self.obs[self._global_step], obs)
        np.copyto(self.actions[self._global_step], action)
        np.copyto(self.rewards[self._global_step], reward)
        np.copyto(self.obs[(self._global_step + 1) % self._storage_size], next_obs)
        np.copyto(self.terminateds[self._global_step], terminated)

        self._global_step = (self._global_step + 1) % self._storage_size
        self._full = self._full or self._global_step == 0

    def sample(self) -> Tuple[th.Tensor]:
        """Sample transitions from the storage.

        Args:
            None.

        Returns:
            Batched samples.
        """
        indices = np.random.randint(
            0,
            self._storage_size if self._full else self._global_step,
            size=self._batch_size,
        )

        obs = th.as_tensor(self.obs[indices], device=self._device).float()
        actions = th.as_tensor(self.actions[indices], device=self._device).float()
        rewards = th.as_tensor(self.rewards[indices], device=self._device).float()
        next_obs = th.as_tensor(
            self.obs[(indices + 1) % self._storage_size], device=self._device
        ).float()
        terminateds = th.as_tensor(
            self.terminateds[indices], device=self._device
        ).float()

        return obs, actions, rewards, terminateds, next_obs
