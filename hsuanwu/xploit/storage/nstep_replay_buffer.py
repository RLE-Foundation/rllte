from collections import defaultdict
from torch.utils.data import IterableDataset
import numpy as np
import random

from hsuanwu.common.typing import *


class NStepReplayBuffer(IterableDataset):
    """
    Replay buffer for off-policy algorithms (N-step returns supported).

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param buffer_size: Max number of element in the buffer.
    :param n_step: The number of transitions to consider when computing n-step returns
    :param discount: The discount factor for future rewards.
    """
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 buffer_size: int,
                 n_step: int = 2,
                 discount: float = 0.99
                 ) -> None:
        self._observation_space = observation_space
        self._action_space = action_space
        self._buffer_size = buffer_size
        self._nstep = n_step
        self._discount = discount

        self._current_episode = defaultdict(list)
        self._episodes = list()

        self._num_transitions = 0
        
    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: float,
            discount: float
            ) -> None:
        
        self._current_episode['observation'].append(observation)
        self._current_episode['action'].append(action)
        self._current_episode['reward'].append(reward)
        self._current_episode['done'].append(done)
        self._current_episode['discount'].append(discount)


        if done:
            eps_len = len(self._current_episode['observation']) - 1
            if (self._num_transitions + eps_len) > self._buffer_size:
                early_eps_len = len(self._episodes[0]['observation']) - 1
                self._episodes.pop(0)
                self._num_transitions -= early_eps_len
            self._num_transitions += eps_len
            self._episodes.append(self._current_episode)
            self._current_episode = defaultdict(list)
        

    def _sample(self) -> Tuple[ndarray]:
        episode = random.choice(self._episodes)
        eps_len = len(episode['observation']) - 1

        idx = np.random.randint(0, eps_len - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount

        return (obs, action, reward, discount, next_obs)

    def __iter__(self) -> Batch:
        while True:
            yield self._sample()