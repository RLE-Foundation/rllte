from collections import defaultdict
import numpy as np
import random

from hsuanwu.common.typing import *


class NStepReplayBuffer:
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
                 buffer_dir: str,
                 n_step: int = 2,
                 discount: float = 0.99
                 ) -> None:
        self._observation_space = observation_space
        self._action_space = action_space
        self._buffer_size = buffer_size
        self._buffer_dir = buffer_dir
        self._nstep = n_step
        self._discount = discount

        self._current_episode = defaultdict(list)
        self._episodes = list()

        self._num_episodes = 0
        self._num_transitions = 0
        
    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: float) -> None:
        
        self._current_episode['observation'].append(observation)
        self._current_episode['action'].append(action)
        self._current_episode['reward'].append(reward)
        self._current_episode['done'].append(done)

        if done:
            self._num_episodes += 1
            eps_len = len(self._current_episode['observation']) - 1
            self._num_transitions += eps_len
            self._current_episode = defaultdict(list)


    def sample(self, batch_size: int) -> Batch:
        batch = [self.sample_single_step() for idx in range(batch_size)]
        return batch
        

    def sample_single_step(self) -> Tuple[ndarray]:
        episode = random.choice(self._episodes)
        eps_len = len(episode['observation']) - 1

        idx = np.random.randint(0, eps_len - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['reward'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    @property
    def get_current_size(self):
        return self._size