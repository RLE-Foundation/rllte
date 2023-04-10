import datetime
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset

from hsuanwu.common.typing import Any, Dict, Path, Tuple
from hsuanwu.xploit.storage.utils import dump_episode, episode_len, load_episode


class ReplayStorage:
    """Storage collected experiences to local files.

    Args:
        replay_dir (Path): save directory.

    Returns:
        Storage instance.
    """

    def __init__(self, replay_dir: Path) -> None:
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._num_episodes = 0
        self._num_transitions = 0

    def _store_episode(self, episode: Dict) -> None:
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        dump_episode(episode, self._replay_dir / eps_fn)

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    @property
    def num_transitions(self) -> int:
        return self._num_transitions

    def add(
        self, obs: Any, action: Any, reward: float, terminated: bool, discount: float
    ) -> None:
        self._current_episode["observation"].append(obs)
        self._current_episode["action"].append(action)
        self._current_episode["reward"].append(np.full((1,), reward, np.float32))
        self._current_episode["terminated"].append(
            np.full((1,), terminated, np.float32)
        )
        self._current_episode["discount"].append(np.full((1,), discount, np.float32))

        if terminated:
            episode = dict()
            for key in self._current_episode.keys():
                episode[key] = np.array(self._current_episode[key])

            # save episode to file
            self._store_episode(episode)
            self._current_episode = defaultdict(list)


class NStepReplayStorage(IterableDataset):
    """Replay storage for off-policy algorithms (N-step returns supported).

    Args:
        storage_size (int): Max number of element in the storage.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): Subprocesses to use for data loading.
        pin_memory (bool): Copy Tensors into device/CUDA pinned memory before returning them.
        n_step (int) The number of transitions to consider when computing n-step returns
        discount (float): The discount factor for future rewards.
        fetch_every (int): Loading interval.
        save_snapshot (bool): Save loaded file or not.

    Returns:
        N-step replay storage.
    """

    def __init__(
        self,
        storage_size: int = 500000,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        n_step: int = 3,
        discount: float = 0.99,
        fetch_every: int = 1000,
        save_snapshot: bool = False,
    ) -> None:
        # set storage
        self._replay_dir = Path.cwd() / "storage"
        self._replay_storage = ReplayStorage(self._replay_dir)
        self._storage_size = storage_size
        self._batch_size = batch_size
        self._num_workers = max(1, num_workers)
        self._pin_memory = pin_memory
        self._n_step = n_step
        self._discount = discount
        self._save_snapshot = save_snapshot

        # setup for single worker
        self._worker_eps_pool = dict()
        self._worker_eps_fn_pool = list()
        self._worker_size = 0
        self._worker_max_size = storage_size // max(1, num_workers)
        self._fetch_every = fetch_every
        self._fetched_samples = fetch_every

    @property
    def get_batch_size(self):
        return self._batch_size

    @property
    def get_num_workers(self):
        return self._num_workers

    @property
    def get_pin_memory(self):
        return self._pin_memory

    def _sample_episode(self) -> Dict:
        eps_fn = random.choice(self._worker_eps_fn_pool)
        return self._worker_eps_pool[eps_fn]

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
        if "discount" in info.keys():
            discount = info['discount'][0]
        elif "discount" in info['final_info'][0].keys():
            discount = info['final_info'][0]['discount']
        else:
            raise ValueError("When using NStepReplayStorage, please put the discount factor in 'info'!")

        self._replay_storage.add(obs, action, reward, terminated, discount)

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
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._worker_eps_pool.keys():
                break
            if fetched_size + eps_len > self._worker_max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def sample(self) -> Tuple:
        """Generate samples.

        Args:
            None.

        Returns:
            Batched samples.
        """
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._fetched_samples += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._n_step + 1) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._n_step - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._n_step):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self.sample()
