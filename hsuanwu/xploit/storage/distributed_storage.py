import torch

from hsuanwu.common.typing import Batch, Device, Tensor, Tuple


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
    ) -> None:
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._device = torch.device(device)
        self._num_steps = num_steps
        self._num_storages = num_storages

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
            policy_logits=dict(size=(num_steps + 1, self._action_dim), dtype=torch.float32),
            baseline=dict(size=(num_steps + 1,), dtype=torch.float32),
            action=dict(size=(num_steps + 1,), dtype=torch.int64),
            episode_win=dict(size=(num_steps + 1,), dtype=torch.int32),
            carried_obj=dict(size=(num_steps + 1,), dtype=torch.int32),
            carried_col=dict(size=(num_steps + 1,), dtype=torch.int32),
            partial_obs=dict(size=(num_steps + 1, 7, 7, 3), dtype=torch.uint8),
            episode_state_count=dict(size=(num_steps + 1,), dtype=torch.float32),
            train_state_count=dict(size=(num_steps + 1,), dtype=torch.float32),
        )
        self.storages = {key: [] for key in specs}

        for _ in range(self._num_storages):
            for key in self.storages:
                self.storages[key].append(torch.empty(**specs[key]).share_memory_())
    
    def add(self, ) -> None:
        """Add sampled transitions into storage.    
        """
    
    def sample(self) -> Batch:
        """Sample transitions from the storage.

        Args:
            None.

        Returns:
            Batched samples.
        """