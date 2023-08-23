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


import warnings
from collections import deque
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.prototype import BaseStorage
from rllte.common.type_alias import PrioritizedReplayBatch


class PrioritizedReplayStorage(BaseStorage):
    """Prioritized replay storage with proportional prioritization for off-policy algorithms.
        Since the storage updates the priorities of the samples based on the TD error, users
        should include the `indices` and `weights` in the returned information of the `.update`
        method of the agent. An example is:
            return {"indices": indices, "weights": weights, ..., "Actor Loss": actor_loss, ...}

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        device (str): Device to convert the data.
        storage_size (int): The capacity of the storage.
        num_envs (int): The number of parallel environments.
        batch_size (int): Batch size of samples.
        alpha (float): Prioritization value.
        beta (float): Importance sampling value.

    Returns:
        Prioritized replay storage.
    """

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 device: str = "cpu",
                 storage_size: int = 1000000,
                 batch_size: int = 1024,
                 num_envs: int = 1,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 ) -> None:
        super().__init__(observation_space, action_space, device, storage_size, batch_size, num_envs)
        # TODO: add support for parallel environments
        warnings.warn("PrioritizedReplayStorage currently does not support parallel environments.") if num_envs != 1 else None
        assert alpha > 0, "The prioritization value `alpha` must be positive!"
        self.alpha = alpha
        self.beta = beta
        self.reset()
    
    def reset(self) -> None:
        """Reset the storage."""
        self.transitions = deque(maxlen=self.storage_size)
        self.priorities = np.zeros((self.storage_size,), dtype=np.float32)
        self.global_step = 0
        super().reset()

    def __len__(self) -> int:
        """Return the number of transitions in storage."""
        return len(self.transitions)

    @property
    def annealing_beta(self) -> float:
        """Linearly increases beta from the initial value to 1 over global training steps."""
        return min(1.0, self.beta + self.global_step * (1.0 - self.beta) / self.storage_size)

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
        transition = (
            observations[0].cpu().numpy(),
            actions[0].cpu().numpy(),
            rewards[0].cpu().numpy(),
            terminateds[0].cpu().numpy(),
            truncateds[0].cpu().numpy(),
            next_observations[0].cpu().numpy(),
        )
        max_prio = self.priorities.max() if self.transitions else 1.0
        self.priorities[self.step] = max_prio
        self.transitions.append(transition)

        self.step = (self.step + 1) % self.storage_size
        self.full = self.full or self.step == 0
        self.global_step += 1

    def sample(self) -> PrioritizedReplayBatch:
        """Sample from the storage."""
        if len(self.transitions) == self.storage_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.step]

        # compute probabilities and sample indices
        probs = priorities**self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.transitions), self.batch_size, p=probs)

        # get samples
        samples = [self.transitions[i] for i in indices]
        weights = (len(self.transitions) * probs[indices]) ** (-self.annealing_beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # unpack
        obs, actions, rewards, terminateds, truncateds, next_obs = zip(*samples)
        obs = np.stack(obs)
        actions = np.stack(actions)
        rewards = np.expand_dims(np.stack(rewards), 1)
        terminateds = np.expand_dims(np.stack(terminateds), 1)
        truncateds = np.expand_dims(np.stack(truncateds), 1)
        next_obs = np.stack(next_obs)

        return PrioritizedReplayBatch(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            rewards=self.to_torch(rewards),
            terminateds=self.to_torch(terminateds),
            truncateds=self.to_torch(truncateds),
            next_observations=self.to_torch(next_obs),
            weights=self.to_torch(weights),
            indices=indices
        )

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
