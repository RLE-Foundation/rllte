import numpy as np

from hsuanwu.common.typing import *

class ReplayBuffer:
    """
    Replay buffer for off-policy algorithms.

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param buffer_size: Max number of element in the buffer.
    """
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 buffer_size: int,
                 ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.buffer_size = buffer_size

        self.observations = np.empty(shape=(self.buffer_size + 1, ) + observation_space.shape, dtype=observation_space.dtype)
        self.actions = np.empty(shape=(self.buffer_size, action_space.shape[0]), dtype=action_space.dtype)
        self.rewards = np.empty(shape=(self.buffer_size, ), dtype=np.float32)
        self.dones = np.empty((self.buffer_size, ), dtype=np.float32)

        self._idx = 0
        self._size = 0

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: float) -> None:

        self.observations[self._idx + 1] = obs
        self.actions[self._idx] = action
        self.rewards[self._idx] = reward
        self.dones[self._idx] = done

        self._idx = (self._idx + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Batch:
        indices = np.random.randint(self._size, size=batch_size)
        return Batch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.observations[indices]
        )

    @property
    def get_current_size(self):
        return self._size