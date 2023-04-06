import threading

import torch

from hsuanwu.common.typing import (
    Batch,
    Device,
    DictConfig,
    List,
    SimpleQueue,
    Storage,
    Tuple,
)


class DistributedStorage:
    """Distributed storage for distributed algorithms like IMPALA.

    Args:
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        obs_shape (Tuple): The data shape of observations.
        action_shape (Tuple): The data shape of actions.
        action_type (str): The type of actions, 'cont' or 'dis'.
        num_steps (int): The sample steps of per rollout.
        num_storages (int): The number of shared-memory storages.

    Returns:
        Vanilla rollout storage.
    """

    def __init__(
        self,
        device: Device,
        obs_shape: Tuple,
        action_shape: Tuple,
        action_type: str,
        num_steps: int = 100,
        num_storages: int = 80,
        batch_size: int = 32,
    ) -> None:
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._device = torch.device(device)
        self._num_steps = num_steps
        self._num_storages = num_storages
        self._batch_size = batch_size

        if action_type == "dis":
            self._action_dim = 1
        elif action_type == "cont":
            self._action_dim = action_shape[0]
        else:
            raise NotImplementedError
        
        specs = dict(
            frame=dict(size=(num_steps + 1, *obs_shape), dtype=torch.uint8),
            reward=dict(size=(num_steps + 1,), dtype=torch.float32),
            done=dict(size=(num_steps + 1,), dtype=torch.bool),
            episode_return=dict(size=(num_steps + 1,), dtype=torch.float32),
            episode_step=dict(size=(num_steps + 1,), dtype=torch.int32),
            last_action=dict(size=(num_steps + 1,), dtype=torch.int64),
            policy_logits=dict(size=(num_steps + 1, action_shape[0]), dtype=torch.float32),
            baseline=dict(size=(num_steps + 1,), dtype=torch.float32),
            action=dict(size=(num_steps + 1,), dtype=torch.int64),
        )

        self.storages = {key: [] for key in specs}
        for _ in range(num_storages):
            for key in self.storages:
                self.storages[key].append(torch.empty(**specs[key]).share_memory_())

    @staticmethod
    def sample(
        device: Device,
        batch_size: int,
        free_queue: SimpleQueue,
        full_queue: SimpleQueue,
        storages: List,
        init_actor_states: List,
        lock=threading.Lock(),
    ) -> Batch:
        """Sample transitions from the storage.

        Args:
            None.

        Returns:
            Batched samples.
        """
        with lock:
            indices = [full_queue.get() for _ in range(batch_size)]
        batch = {
            key: torch.stack([storages[key][m] for m in indices], dim=1)
            for key in storages
        }

        init_actor_state = (
            torch.cat(ts, dim=1)
            for ts in zip(*[init_actor_states[m] for m in indices])
        )

        for i in indices:
            free_queue.put(i)

        batch = {
            k: t.to(device=torch.device(device), non_blocking=True)
            for k, t in batch.items()
        }
        return batch, init_actor_state
