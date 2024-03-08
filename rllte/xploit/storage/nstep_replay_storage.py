from typing import Any, Dict, Tuple, Deque
from collections import deque

import gym
import numpy as np
import torch as th

from rllte.common.prototype import BaseStorage
from rllte.common.type_alias import VanillaReplayBatch

class NStepReplayStorage(BaseStorage):
    """N-step replay storage.
        Implemented based on: https://colab.research.google.com/github/Curt-Park/rainbow-is-all-you-need/blob/master/07.n_step_learning.ipynb
        
    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        device (str): Device to convert the data.
        storage_size (int): The capacity of the storage.
        num_envs (int): The number of parallel environments.
        batch_size (int): Batch size of samples.
        n_step (int): Number of steps for the n-step transition.
        gamma (float): Discount factor.

    Returns:
        N-step replay storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        storage_size: int = 1000000,
        batch_size: int = 1024,
        num_envs: int = 1,
        n_step: int = 3,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(observation_space, action_space, device, storage_size, batch_size, num_envs)
        self.storage_size = max(storage_size // num_envs, 1)
        self.reset()
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def reset(self) -> None:
        """Reset the storage."""
        self.observations = np.empty((self.storage_size, self.num_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.next_observations = np.empty(
            (self.storage_size, self.num_envs, *self.obs_shape), dtype=self.observation_space.dtype
        )
        self.actions = np.empty((self.storage_size, self.num_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.empty((self.storage_size, self.num_envs), dtype=np.float32)
        self.terminateds = np.empty((self.storage_size, self.num_envs), dtype=np.float32)
        self.truncateds = np.empty((self.storage_size, self.num_envs), dtype=np.float32)
        super().reset()

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
        for i in range(len(observations)):
            self.n_step_buffer.append((
                observations[i], actions[i], rewards[i], terminateds[i], truncateds[i], next_observations[i]
            ))
            if len(self.n_step_buffer) == self.n_step:
                obs, act, rew, term, trunc, next_obs = self._get_n_step_info()
                self._store_transition(obs, act, rew, term, trunc, next_obs)

    def _get_n_step_info(self) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, Dict[str, Any], th.Tensor]:
        """Calculate n-step information."""
        rew, next_obs, term, trunc = 0, None, False, False
        for i in range(self.n_step):
            obs, act, r, t, tr, next_obs = self.n_step_buffer[i]
            rew += r * (self.gamma ** i)
            if i == self.n_step - 1:
                next_obs = next_obs
                term = t
                trunc = tr
        return obs, act, rew, term, trunc, next_obs

    def _store_transition(self, obs, act, rew, term, trunc, next_obs):
        """Store a single transition."""
        np.copyto(self.observations[self.step], obs.cpu().numpy())
        np.copyto(self.actions[self.step], act.cpu().numpy())
        np.copyto(self.rewards[self.step], rew.cpu().numpy())
        np.copyto(self.next_observations[self.step], next_obs.cpu().numpy())
        np.copyto(self.terminateds[self.step], term.cpu().numpy())
        np.copyto(self.truncateds[self.step], trunc.cpu().numpy())
        self.step = (self.step + 1) % self.storage_size
        self.full = self.full or self.step == 0

    def sample(self) -> VanillaReplayBatch:
        """Sample from the storage."""
        if self.full:
            batch_indices = (np.random.randint(1, self.storage_size, size=self.batch_size) + self.step) % self.storage_size
        else:
            batch_indices = np.random.randint(0, self.step, size=self.batch_size)
        env_indices = np.random.randint(0, self.num_envs, size=(self.batch_size,))

        obs = self.observations[batch_indices, env_indices, :]
        actions = self.actions[batch_indices, env_indices, :]
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, 1)
        terminateds = self.terminateds[batch_indices, env_indices].reshape(-1, 1)
        truncateds = self.truncateds[batch_indices, env_indices].reshape(-1, 1)
        next_obs = self.next_observations[batch_indices, env_indices, :]

        return VanillaReplayBatch(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            rewards=self.to_torch(rewards),
            terminateds=self.to_torch(terminateds),
            truncateds=self.to_torch(truncateds),
            next_observations=self.to_torch(next_obs),
        )
        
    def update(self, *args) -> None:
        """Update the storage if necessary."""
        return None