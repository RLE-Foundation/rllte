from collections import defaultdict
import numpy as np
import datetime
import io

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
        self.observation_space = observation_space
        self.action_space = action_space
        self.buffer_size = buffer_size
        self.buffer_dir = buffer_dir
        self.n_step = n_step
        self.discount = discount

        self._current_episode = defaultdict(list)

        self._num_episodes = 0
        self._num_transitions = 0

    def _load_episode(self, file):
        with file.open('rb') as f:
            episode = np.load(f)
            episode = {key: episode[key] for key in episode.keys()}
            return episode

    def _save_episode(self):
        eps_idx = self._num_episodes
        eps_len = len(self._current_episode['observation']) - 1
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_file = f'{ts}_{eps_idx}_{eps_len}.npz'

        with io.BytesIO() as bs:
            np.savez_compressed(bs, **self._current_episode)
            bs.seek(0)
            with eps_file.open('wb') as f:
                f.write(bs.read())
        
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
            self._save_episode()
            self._current_episode = defaultdict(list)
    
    def sample(self, batch_size: int) -> Batch:
        pass

    @property
    def get_current_size(self):
        return self._size