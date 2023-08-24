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
from typing import Any, Dict

import gymnasium as gym
import torch as th
from torch import multiprocessing as mp

from rllte.common.prototype import BaseStorage
from rllte.common.preprocessing import is_image_space

class VanillaDistributedStorage(BaseStorage):
    """Vanilla distributed storage for distributed algorithms like IMPALA.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        storage_size (int): The capacity of the storage. Here it refers to the length of per rollout.
        num_storages (int): The number of shared-memory storages.
        num_envs (int): The number of parallel environments.
        batch_size (int): The batch size.

    Returns:
        Vanilla distributed storage.
    """

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 device: str = "cpu",
                 storage_size: int = 100,
                 num_storages: int = 80,
                 num_envs: int = 45,
                 batch_size: int = 32
                 ) -> None:
        super().__init__(observation_space, action_space, device, storage_size, batch_size, num_envs)
        self.num_storages = num_storages
        self.reset()

    def reset(self) -> None:
        """Reset the storage."""
        obs_dtype = th.uint8 if is_image_space(self.observation_space) else th.float32
        action_dtype = th.float32 if self.action_type is "Box" else th.int64
        policy_outputs_dim = self.policy_action_dim * 2 if self.action_type is "Box" else self.policy_action_dim
        specs = dict(
            observations=dict(size=(self.storage_size + 1, *self.obs_shape), dtype=obs_dtype),
            actions=dict(size=(self.storage_size + 1, self.action_dim), dtype=action_dtype),
            rewards=dict(size=(self.storage_size + 1, ), dtype=th.float32),
            terminateds=dict(size=(self.storage_size + 1, ), dtype=th.bool),
            truncateds=dict(size=(self.storage_size + 1, ), dtype=th.bool),
            episode_returns=dict(size=(self.storage_size + 1, ), dtype=th.float32),
            episode_steps=dict(size=(self.storage_size + 1, ), dtype=th.int32),
            last_actions=dict(size=(self.storage_size + 1, self.action_dim), dtype=action_dtype),
            policy_outputs=dict(size=(self.storage_size + 1, policy_outputs_dim), dtype=th.float32),
            baselines=dict(size=(self.storage_size + 1, ), dtype=th.float32)
        )

        # create memory-shared storages
        self.storages = {key: [] for key in specs}
        for _ in range(self.num_storages):
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

    def sample(self, 
               free_queue: mp.SimpleQueue, 
               full_queue: mp.SimpleQueue, 
               lock=threading.Lock()  # noqa B008
               ) -> Dict[str, th.Tensor]:
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

    def update(self, *args, **kwargs) -> None:
        """Update the storage"""
        return None
