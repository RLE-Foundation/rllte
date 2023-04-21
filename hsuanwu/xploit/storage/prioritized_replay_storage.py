from typing import Tuple, Union, Any
import gymnasium as gym
from omegaconf import DictConfig
import numpy as np
import torch as th
from collections import deque

from hsuanwu.xploit.storage.base import BaseStorage


class PrioritizedReplayStorage(BaseStorage):
    """Prioritized replay storage with proportional prioritization for off-policy algorithms.

    Args:
        obs_space (Space or DictConfig): The observation space of environment. When invoked by Hydra, 
            'obs_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like 
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        storage_size (int): Max number of element in the buffer.
        batch_size (int): Batch size of samples.
        alpha (float): The alpha coefficient.
        beta (float): The beta coefficient.

    Returns:
        Prioritized replay storage.
    """
    def __init__(
        self,
        obs_space: Union[gym.Space, DictConfig],
        action_space: Union[gym.Space, DictConfig],
        device: th.device = 'cpu',
        storage_size: int = 1000000,
        batch_size: int = 1024,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        super().__init__(obs_space, action_space, device)
        self._storage_size = storage_size
        self._batch_size = batch_size
        assert alpha > 0, "The prioritization value 'alpha' must be positive!"
        self._alpha = alpha
        self._beta = beta
        self._storage = deque(maxlen=storage_size)
        self._priorities = np.zeros((storage_size, ), dtype=np.float32)
        self._position = 0
    
    def __len__(self):
        return len(self._storage)
    
    def annealing_beta(self, step: int) -> float:
        """Linearly increases beta from the initial value to 1 over global training steps.

        Args:
            step (int): The global training step.
        
        Returns:
            Beta value.
        """
        return min(1.0, self._beta + step * (1.0 - self._beta) / self._storage_size)
    
    def add(self,
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
        transition = (obs, action, reward, terminated, next_obs)
        max_prio = self._priorities.max() if self._storage else 1.0
        self._priorities[self._position] = max_prio
        self._storage.append(transition)
        self._position = (self._position + 1) % self._storage_size

    def sample(self, step: int) -> Tuple[th.Tensor, ...]:
        """Sample from the storage.

        Args:
            step (int): Global training step.

        Returns:
            Batched samples.
        """
        if len(self._storage) == self._storage_size:
            priorities = self._priorities
        else:
            priorities = self._priorities[:self._position]
        

        probs = priorities ** self._alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self._storage), self._batch_size, p=probs)

        samples = [self._storage[i] for i in indices]
        weights = (len(self._storage) * probs[indices]) ** (-self.annealing_beta(step))
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        obs, actions, rewards, terminateds, next_obs = zip(*samples)
        obs = np.stack(obs)
        actions = np.stack(actions)
        rewards = np.expand_dims(np.stack(rewards), 1)
        terminateds = np.expand_dims(np.stack(terminateds), 1)
        next_obs = np.stack(next_obs)

        obs = th.as_tensor(obs, device=self._device).float()
        actions = th.as_tensor(actions, device=self._device).float()
        rewards = th.as_tensor(rewards, device=self._device).float()
        next_obs = th.as_tensor(next_obs, device=self._device).float()
        terminateds = th.as_tensor(terminateds, device=self._device).float()
        weights = th.as_tensor(weights, device=self._device).float()

        return indices, obs, actions, rewards, terminateds, next_obs, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update the priorities.

        Args:
            indices (NdArray): The indices of current batch data.
            priorities (NdArray): The priorities of current batch data.
        
        Returns:
            None.
        """
        for i, priority in zip(indices, priorities):
            self._priorities[i] = abs(priority)
