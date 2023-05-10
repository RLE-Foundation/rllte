import threading
from typing import Any, Dict, Generator, List, Tuple, Union

import gymnasium as gym
import torch as th
from omegaconf import DictConfig

from hsuanwu.xploit.storage.base import BaseStorage


class DistributedStorage(BaseStorage):
    """Distributed storage for distributed algorithms like IMPALA.

    Args:
        observation_space (Space or DictConfig): The observation space of environment. When invoked by Hydra,
            'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like
            {"shape": action_space.shape, "n": action_space.n, "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        num_steps (int): The sample steps of per rollout.
        num_storages (int): The number of shared-memory storages.
        batch_size (int): The batch size.

    Returns:
        Vanilla rollout storage.
    """

    def __init__(
        self,
        observation_space: Union[gym.Space, DictConfig],
        action_space: Union[gym.Space, DictConfig],
        device: str = "cpu",
        num_steps: int = 100,
        num_storages: int = 80,
        batch_size: int = 32,
    ) -> None:
        super().__init__(observation_space, action_space, device)
        self._num_steps = num_steps
        self._num_storages = num_storages
        self._batch_size = batch_size

        if self._action_type == "Discrete":
            self._action_dim = 1
            policy_outputs_dim = self._action_shape[0]

            specs = dict(
                obs=dict(size=(num_steps + 1, *self._obs_shape), dtype=th.uint8),
                reward=dict(size=(num_steps + 1,), dtype=th.float32),
                terminated=dict(size=(num_steps + 1,), dtype=th.bool),
                truncated=dict(size=(num_steps + 1,), dtype=th.bool),
                episode_return=dict(size=(num_steps + 1,), dtype=th.float32),
                episode_step=dict(size=(num_steps + 1,), dtype=th.int32),
                last_action=dict(size=(num_steps + 1,), dtype=th.int64),
                policy_outputs=dict(size=(num_steps + 1, policy_outputs_dim), dtype=th.float32),
                baseline=dict(size=(num_steps + 1,), dtype=th.float32),
                action=dict(size=(num_steps + 1,), dtype=th.int64),
            )

        elif self._action_type == "Box":
            self._action_dim = self._action_shape[0]
            policy_outputs_dim = self._action_shape[0] * 2

            specs = dict(
                obs=dict(size=(num_steps + 1, *self._obs_shape), dtype=th.uint8),
                reward=dict(size=(num_steps + 1,), dtype=th.float32),
                terminated=dict(size=(num_steps + 1,), dtype=th.bool),
                truncated=dict(size=(num_steps + 1,), dtype=th.bool),
                episode_return=dict(size=(num_steps + 1,), dtype=th.float32),
                episode_step=dict(size=(num_steps + 1,), dtype=th.int32),
                last_action=dict(size=(num_steps + 1, self._action_dim), dtype=th.float32),
                policy_outputs=dict(size=(num_steps + 1, policy_outputs_dim), dtype=th.float32),
                baseline=dict(size=(num_steps + 1,), dtype=th.float32),
                action=dict(size=(num_steps + 1, self._action_dim), dtype=th.float32),
            )
        else:
            raise NotImplementedError

        self.storages = {key: [] for key in specs}
        for _ in range(num_storages):
            for key in self.storages:
                self.storages[key].append(th.empty(**specs[key]).share_memory_())

    def add(self, *args) -> None:
        """Add sampled transitions into storage."""

    @staticmethod
    def sample(
        device: th.device,
        batch_size: int,
        free_queue: th.multiprocessing.SimpleQueue,
        full_queue: th.multiprocessing.SimpleQueue,
        storages: List,
        init_actor_state_storages: List,
        lock=threading.Lock(),  # noqa B008
    ) -> Tuple[Dict, Generator[Any, Any, None]]:
        """Sample transitions from the storage.

        Args:
            device (Device): Device (cpu, cuda, ...) on which the code should be run.
            batch_size (int): The batch size.
            free_queue (Queue): Free queue for communication.
            full_queue (Queue): Full queue for communication.
            storages (List[Storage]): A list of shared storages.
            init_actor_state_storages: (List[Tensor]): Initial states for LSTM.
            lock (Lock): Thread lock.

        Returns:
            Batched samples.
        """
        with lock:
            indices = [full_queue.get() for _ in range(batch_size)]
        batch = {key: th.stack([storages[key][i] for i in indices], dim=1) for key in storages}

        init_actor_states = (th.cat(ts, dim=1) for ts in zip(*[init_actor_state_storages[i] for i in indices]))

        for i in indices:
            free_queue.put(i)

        batch = {key: tensor.to(device=th.device(device), non_blocking=True) for key, tensor in batch.items()}
        return batch, init_actor_states

    def update(self, *args) -> None:
        """Update the storage"""
