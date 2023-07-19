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
from torch import multiprocessing as mp

import gymnasium as gym
import torch as th

from rllte.common.base_storage import BaseStorage


class VanillaDistributedStorage(BaseStorage):
    """Vanilla distributed storage for distributed algorithms like IMPALA.

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

        # data containers
        ###########################################################################################################
        if self.action_type == "Discrete":
            specs = dict(
                observation=dict(size=(num_steps + 1, *self.obs_shape), dtype=th.uint8),
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
                observation=dict(size=(num_steps + 1, *self.obs_shape), dtype=th.uint8),
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
            raise NotImplementedError(f"Unsupported action space {self.action_type}.")
        ###########################################################################################################

        # Create memory-shared storages.
        self.storages = {key: [] for key in specs}
        for _ in range(num_storages):
            for key in self.storages:
                self.storages[key].append(th.empty(**specs[key]).share_memory_())

    def add(self, 
            idx: int,
            timestep: int,
            actor_output: Dict[str, Any],
            env_output: Dict[str, Any],
            ) -> None:
        """Add sampled transitions into storage.

        Args:
            idx (int): The index of storage.
            timestep (int): The timestep of rollout.
            actor_output (Dict): Actor output.
            env_output (Dict): Environment output.
        
        Returns:
            None
        """
        for key in env_output:
            self.storages[key][idx][timestep, ...] = env_output[key]
        for key in actor_output:
            self.storages[key][idx][timestep, ...] = actor_output[key]

    def sample(self, # noqa B008
               free_queue: mp.SimpleQueue, 
               full_queue: mp.SimpleQueue, 
               lock=threading.Lock()
               ) -> Tuple[Dict, Generator[Any, Any, None]]:
        """Sample transitions from the storage.

        Args:
            free_queue (Queue): Free queue for communication.
            full_queue (Queue): Full queue for communication.
            lock (Lock): Thread lock.

        Returns:
            Batched samples.
        """
        with lock:
            indices = [full_queue.get() for _ in range(self.batch_size)]
        batch = {key: th.stack([self.storages[key][i] for i in indices], dim=1) for key in self.storages}

        # serialize the data
        for i in indices:
            free_queue.put(i)

        batch = {key: tensor.to(device=self.device, non_blocking=True) for key, tensor in batch.items()}
        return batch

    def update(self, *args) -> None:
        """Update the storage"""
        return None
