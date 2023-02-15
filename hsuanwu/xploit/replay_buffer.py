from gym import spaces
import numpy as np

from hsuanwu.common.typing import *


class ReplayBuffer:
    """
    Replay buffer for off-policy algorithms

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param n_envs: Number of parallel environments
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1
        ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.buffer_size = max(buffer_size // n_envs, 1)

        self.observations = np.empty(shape=(self.buffer_size + 1, self.n_envs) + observation_space.shape, dtype=observation_space.dtype)
        self.actions = np.empty(shape=(self.buffer_size, self.n_envs, action_space.shape[0]), dtype=action_space.dtype)
        self.rewards = np.empty(shape=(self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.masks = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        # self.next_observations = np.empty(shape=(self.buffer_size, self.n_envs) + observation_space.shape, dtype=observation_space.dtype)

        self._idx = 0
        self._size = 0

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            mask: float,
            done: float) -> None:

        self.observations[self.idx + 1] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.masks[self.idx] = mask
        self.dones[self.idx] = done

        self._idx = (self._idx + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Batch:
        indices = np.random.randint(self._size, size=batch_size)
        return Batch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            masks=self.masks[indices],
            next_observations=self.observations[indices]
        )

    @property
    def get_current_size(self):
        return self._size