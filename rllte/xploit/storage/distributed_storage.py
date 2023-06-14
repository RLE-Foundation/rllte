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

import threading
from typing import Any, Dict, Generator, List, Tuple

import gymnasium as gym
import torch as th

from rllte.common.base_storage import BaseStorage


class DistributedStorage(BaseStorage):
    """Distributed storage for distributed algorithms like IMPALA.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        num_steps (int): The sample steps of per rollout.
        num_storages (int): The number of shared-memory storages.
        batch_size (int): The batch size.

    Returns:
        Vanilla rollout storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        num_steps: int = 100,
        num_storages: int = 80,
        batch_size: int = 32,
    ) -> None:
        super().__init__(observation_space, action_space, device)
        self.num_steps = num_steps
        self.num_storages = num_storages
        self.batch_size = batch_size

        if self.action_type == "Discrete":
            specs = dict(
                obs=dict(size=(num_steps + 1, *self.obs_shape), dtype=th.uint8),
                reward=dict(size=(num_steps + 1,), dtype=th.float32),
                terminated=dict(size=(num_steps + 1,), dtype=th.bool),
                truncated=dict(size=(num_steps + 1,), dtype=th.bool),
                episode_return=dict(size=(num_steps + 1,), dtype=th.float32),
                episode_step=dict(size=(num_steps + 1,), dtype=th.int32),
                last_action=dict(size=(num_steps + 1,), dtype=th.int64),
                policy_outputs=dict(size=(num_steps + 1, self.action_dim), dtype=th.float32),
                baseline=dict(size=(num_steps + 1,), dtype=th.float32),
                action=dict(size=(num_steps + 1,), dtype=th.int64),
            )

        elif self.action_type == "Box":
            specs = dict(
                obs=dict(size=(num_steps + 1, *self.obs_shape), dtype=th.uint8),
                reward=dict(size=(num_steps + 1,), dtype=th.float32),
                terminated=dict(size=(num_steps + 1,), dtype=th.bool),
                truncated=dict(size=(num_steps + 1,), dtype=th.bool),
                episode_return=dict(size=(num_steps + 1,), dtype=th.float32),
                episode_step=dict(size=(num_steps + 1,), dtype=th.int32),
                last_action=dict(size=(num_steps + 1, self.action_dim), dtype=th.float32),
                policy_outputs=dict(size=(num_steps + 1, self.action_dim * 2), dtype=th.float32),
                baseline=dict(size=(num_steps + 1,), dtype=th.float32),
                action=dict(size=(num_steps + 1, self.action_dim), dtype=th.float32),
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
        storages: Dict[str, list],
        init_actor_state_storages: List,
        lock=threading.Lock(),  # noqa B008
    ) -> Tuple[Dict, Generator[Any, Any, None]]:
        """Sample transitions from the storage.

        Args:
            device (Device): Device (cpu, cuda, ...) on which the code should be run.
            batch_size (int): The batch size.
            free_queue (Queue): Free queue for communication.
            full_queue (Queue): Full queue for communication.
            storages (Dict[str, list]): A Dict of shared storages.
            init_actor_state_storages: (List[th.Tensor]): Initial states for LSTM.
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
        raise NotImplementedError
