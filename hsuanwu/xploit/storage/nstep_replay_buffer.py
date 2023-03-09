from torch.utils.data import IterableDataset

import numpy as np
import traceback
import random

from hsuanwu.common.typing import *
from hsuanwu.xploit.storage.vanilla_replay_buffer import ReplayBufferStorage
from hsuanwu.xploit.storage.utils import episode_len, load_episode

class NStepReplayBuffer(IterableDataset):
    """Replay buffer for off-policy algorithms (N-step returns supported).

    Args:
        buffer_size: Max number of element in the buffer.
        num_workers: Subprocesses to use for data loading.
        n_step: The number of transitions to consider when computing n-step returns
        discount: The discount factor for future rewards.
        fetch_every: Loading interval.
        save_snapshot: Save loaded file or not.

    Returns:
        N-step replay buffer.
    """
    def __init__(self,
                 buffer_size: int,
                 num_workers: int,
                 n_step: int = 2,
                 discount: float = 0.99,
                 fetch_every: int = 1000,
                 save_snapshot: bool = False
                 ) -> None:
        # set storage
        self._replay_dir = Path.cwd() / 'buffer'
        self._replay_storage = ReplayBufferStorage(self._replay_dir)
        self._buffer_size = buffer_size
        self._num_workers = max(1, num_workers)
        self._n_step = n_step
        self._discount = discount
        self._save_snapshot = save_snapshot

        # setup for single worker
        self._worker_eps_pool = dict()
        self._worker_eps_fn_pool = list()
        self._worker_size = 0
        self._worker_max_size = buffer_size // max(1, num_workers)
        self._fetch_every = fetch_every
        self._fetched_samples = fetch_every


    def _sample_episode(self) -> Dict:
        eps_fn = random.choice(self._worker_eps_fn_pool)
        return self._worker_eps_pool[eps_fn]


    def add(self,
            observation: Any,
            action: Any,
            reward: float,
            done: float,
            info: Any) -> None:
        self._replay_storage.add(observation, action, reward, done, info, use_discount=True)


    def _store_episode(self, eps_fn: Path) -> bool:
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._worker_size > self._worker_max_size:
            early_eps_fn = self._worker_eps_fn_pool.pop(0)
            early_eps = self._worker_eps_pool.pop(early_eps_fn)
            self._worker_size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._worker_eps_fn_pool.append(eps_fn)
        self._worker_eps_fn_pool.sort()
        self._worker_eps_pool[eps_fn] = episode
        self._worker_size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self) -> None:
        if self._fetched_samples < self._fetch_every:
            return
        self._fetched_samples = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._worker_eps_pool.keys():
                break
            if fetched_size + eps_len > self._worker_max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self) -> Tuple:
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._fetched_samples += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._n_step + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._n_step - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._n_step):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()