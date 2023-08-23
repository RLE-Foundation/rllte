# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import datetime
import random
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from torch.utils.data import IterableDataset

from rllte.common.prototype import BaseStorage
from rllte.common.type_alias import NStepReplayBatch
from rllte.xploit.storage.utils import episode_len, load_episode, save_episode, worker_init_fn


class ReplayStorage:
    """Replay storage for storing transitions.
        Implemented based on: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py

    Args:
        replay_dir (Path): Directory to store replay data.

    Returns:
        Replay storage.
    """

    def __init__(self, replay_dir: Path) -> None:
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self) -> int:
        return self._num_transitions

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        infos: Dict[str, Any],
        next_obs: np.ndarray,
    ) -> None:
        """Add a new transition to the storage.

        Args:
            obs (np.ndarray): Observation.
            action (np.ndarray): Action.
            reward (float): Reward.
            terminated (bool): Termination flag.
            truncated (bool): Truncation flag.
            infos (Dict): Additional information.
            next_obs (np.ndarray): Next observation.

        Returns:
            None.
        """
        self._current_episode["observation"].append(obs)
        self._current_episode["action"].append(action)
        self._current_episode["reward"].append(np.full((1,), reward, np.float32))
        self._current_episode["terminated"].append(np.full((1,), terminated, np.float32))
        self._current_episode["truncated"].append(np.full((1,), truncated, np.float32))
        self._current_episode["discount"].append(np.full((1,), 1.0, np.float32))

        if terminated or truncated:
            # final next observation
            self._current_episode["observation"].append(infos["final_observation"][0])
            episode = dict()
            for key in self._current_episode.keys():
                episode[key] = np.array(self._current_episode[key])
            # save episode to file
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self) -> None:
        """Preload replay data from disk."""
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode: Dict[str, np.ndarray]) -> None:
        """Store an episode to disk.

        Args:
            episode (Dict[str, np.ndarray]): Episode to be stored.

        Returns:
            None.
        """
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayStorageDataset(IterableDataset):
    """Iterable dataset for replay storage.
        Implemented based on: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py

    Args:
        replay_dir (Path): Directory to store replay data.
        max_size (int): Max number of element in the storage.
        num_workers (int): Subprocesses to use for data loading.
        nstep (int) The number of transitions to consider when computing n-step returns
        discount (float): The discount factor for future rewards.
        fetch_every (int): Loading interval.
        save_snapshot (bool): Save loaded file or not.

    Returns:
        Replay storage dataset.
    """

    def __init__(
        self,
        replay_dir: Path,
        max_size: int,
        num_workers: int,
        nstep: int,
        discount: float,
        fetch_every: int,
        save_snapshot: bool,
    ) -> None:
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self) -> Dict[str, np.ndarray]:
        """Sample an episode from the storage."""
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn: Path) -> bool:
        """Store an episode to the storage."""
        try:
            episode = load_episode(eps_fn)
        except Exception:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self) -> None:
        """Try to fetch new episodes from disk."""
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = th.utils.data.get_worker_info().id
        except Exception:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = (int(x) for x in eps_fn.stem.split("_")[1:])
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self) -> Tuple[np.ndarray, ...]:
        """Sample a transition from the storage."""
        try:
            self._try_fetch()
        except Exception:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - self._nstep)
        obs = episode["observation"][idx]
        actions = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep]
        rewards = np.zeros_like(episode["reward"][idx])
        discounts = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            rewards += discounts * step_reward
            discounts *= episode["discount"][idx + i] * self._discount

        return obs, actions, rewards, discounts, next_obs

    def __iter__(self) -> Iterator:
        while True:
            yield self._sample()


class NStepReplayStorage(BaseStorage):
    """N-step replay storage.
        Implemented based on: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        device (str): Device to convert replay data.
        storage_size (int): Max number of element in the storage.
        num_envs (int): The number of parallel environments.
        batch_size (int): Batch size of samples.
        num_workers (int): Subprocesses to use for data loading.
        pin_memory (bool): Pin memory or not.
        nstep (int): The number of transitions to consider when computing n-step returns
        discount (float): The discount factor for future rewards.
        fetch_every (int): Loading interval.
        save_snapshot (bool): Save loaded file or not.

    Returns:
        N-step replay storage.
    """

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 device: str = "cpu",
                 storage_size: int = 1000000,
                 num_envs: int = 1,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 n_step: int = 3,
                 discount: float = 0.99,
                 fetch_every: int = 1000,
                 save_snapshot: bool = False
                 ) -> None:
        super().__init__(observation_space, action_space, device, storage_size, batch_size, num_envs)
        warnings.warn("NStepReplayStorage currently does not support parallel environments.") if num_envs != 1 else None
        # build storage
        self.replay_dir = Path.cwd() / "storage"
        self.replay_storage = ReplayStorage(self.replay_dir)
        max_size_per_worker = storage_size // max(1, num_workers)
        self.dataset = ReplayStorageDataset(
            replay_dir=self.replay_dir,
            max_size=max_size_per_worker,
            num_workers=num_workers,
            nstep=n_step,
            discount=discount,
            fetch_every=fetch_every,
            save_snapshot=save_snapshot,
        )
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.reset()
    
    def reset(self) -> None:
        """Reset the storage."""
        self.replay_loader = th.utils.data.DataLoader(self.dataset,
                                                      batch_size=self.batch_size,
                                                      num_workers=self.num_workers, 
                                                      pin_memory=self.pin_memory,
                                                      worker_init_fn=worker_init_fn
                                                      )
        self._replay_iter = None

    def add(self,
            observations: th.Tensor,
            actions: th.Tensor,
            rewards: th.Tensor,
            terminateds: th.Tensor,
            truncateds: th.Tensor,
            infos: Dict[str, Any],
            next_observations: th.Tensor
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
        # TODO: add parallel env support
        self.replay_storage.add(
            obs=observations[0].cpu().numpy(),
            action=actions[0].cpu().numpy(),
            reward=rewards[0].cpu().numpy(),
            terminated=terminateds[0].cpu().numpy(),
            truncated=truncateds[0].cpu().numpy(),
            infos=infos,
            next_obs=next_observations[0].cpu().numpy(),
        )

    @property
    def replay_iter(self) -> Iterator:
        """Create iterable dataloader."""
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def sample(self) -> NStepReplayBatch:
        """Sample from the storage."""
        # to device
        obs, actions, rewards, discounts, next_obs = next(self.replay_iter)

        return NStepReplayBatch(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            rewards=self.to_torch(rewards),
            discounts=self.to_torch(discounts),
            next_observations=self.to_torch(next_obs),
        )

    def update(self, *args) -> None:
        """Update the storage if necessary."""
        return None
