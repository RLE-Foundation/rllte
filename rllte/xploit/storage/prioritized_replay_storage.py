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


from collections import deque
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.base_storage import BaseStorage


class PrioritizedReplayStorage(BaseStorage):
    """Prioritized replay storage with proportional prioritization for off-policy algorithms.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        device (str): Device to store the data.
        storage_size (int): Storage size.
        batch_size (int): Batch size.
        alpha (float): Prioritization value.
        beta (float): Importance sampling value.

    Returns:
        Prioritized replay storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        storage_size: int = 1000000,
        batch_size: int = 1024,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        super().__init__(observation_space, action_space, device)
        self.storage_size = storage_size
        self.batch_size = batch_size
        assert alpha > 0, "The prioritization value 'alpha' must be positive!"
        self.alpha = alpha
        self.beta = beta
        self.storage = deque(maxlen=storage_size)
        self.priorities = np.zeros((storage_size,), dtype=np.float32)
        self.position = 0

    def __len__(self) -> int:
        return len(self.storage)

    def annealing_beta(self, step: int) -> float:
        """Linearly increases beta from the initial value to 1 over global training steps.

        Args:
            step (int): The global training step.

        Returns:
            Beta value.
        """
        return min(1.0, self.beta + step * (1.0 - self.beta) / self.storage_size)

    def add(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        terminated: th.Tensor,
        truncated: th.Tensor,
        info: th.Tensor,
        next_obs: th.Tensor,
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            obs (th.Tensor): Observation.
            action (th.Tensor): Action.
            reward (th.Tensor): Reward.
            terminated (th.Tensor): Termination flag.
            truncated (th.Tensor): Truncation flag.
            info (th.Tensor): Additional information.
            next_obs (th.Tensor): Next observation.

        Returns:
            None.
        """
        # TODO: add parallel env support
        transition = (
            obs[0].cpu().numpy(),
            action[0].cpu().numpy(),
            reward[0].cpu().numpy(),
            terminated[0].cpu().numpy(),
            truncated[0].cpu().numpy(),
            next_obs[0].cpu().numpy(),
        )
        max_prio = self.priorities.max() if self.storage else 1.0
        self.priorities[self.position] = max_prio
        self.storage.append(transition)
        self.position = (self.position + 1) % self.storage_size

    def sample(self, step: int) -> Tuple[th.Tensor, ...]:
        """Sample from the storage.

        Args:
            step (int): Global training step.

        Returns:
            Batched samples.
        """
        if len(self.storage) == self.storage_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.position]

        probs = priorities**self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.storage), self.batch_size, p=probs)

        samples = [self.storage[i] for i in indices]
        weights = (len(self.storage) * probs[indices]) ** (-self.annealing_beta(step))
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        obs, actions, rewards, terminateds, truncateds, next_obs = zip(*samples)
        obs = np.stack(obs)
        actions = np.stack(actions)
        rewards = np.expand_dims(np.stack(rewards), 1)
        terminateds = np.expand_dims(np.stack(terminateds), 1)
        truncateds = np.expand_dims(np.stack(truncateds), 1)
        next_obs = np.stack(next_obs)

        obs = th.as_tensor(obs, device=self.device).float()
        actions = th.as_tensor(actions, device=self.device).float()
        rewards = th.as_tensor(rewards, device=self.device).float()
        next_obs = th.as_tensor(next_obs, device=self.device).float()
        terminateds = th.as_tensor(terminateds, device=self.device).float()
        truncateds = th.as_tensor(truncateds, device=self.device).float()
        weights = th.as_tensor(weights, device=self.device).float()

        return indices, obs, actions, rewards, terminateds, truncateds, next_obs, weights

    def update(self, metrics: Dict) -> None:
        """Update the priorities.

        Args:
            metrics (Dict): Training metrics from agent to udpate the priorities:
                indices (np.ndarray): The indices of current batch data.
                priorities (np.ndarray): The priorities of current batch data.

        Returns:
            None.
        """
        if "indices" in metrics and "priorities" in metrics:
            for i, priority in zip(metrics["indices"], metrics["priorities"]):
                self.priorities[i] = abs(priority)
