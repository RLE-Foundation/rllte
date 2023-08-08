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
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch as th
import warnings

from rllte.common.base_storage import BaseStorage, PrioritizedReplayBatch

class PrioritizedReplayStorage(BaseStorage):
    """Prioritized replay storage with proportional prioritization for off-policy algorithms.
        Since the storage updates the priorities of the samples based on the TD error, users 
        should include the `indices` and `weights` in the returned information of the `.update`
        method of the agent. An example is:
            return {"indices": indices, "weights": weights, ..., "Actor Loss": actor_loss, ...}

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        device (str): Device to store the data.
        storage_size (int): Storage size.
        num_envs (int): The number of parallel environments.
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
        num_envs: int = 1,
        batch_size: int = 1024,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        super().__init__(observation_space, action_space, device)
        # TODO: add support for parallel environments
        warnings.warn("NStepReplayStorage currently does not support parallel environments.") if num_envs != 1 else None
        self.storage_size = storage_size
        self.num_envs = num_envs
        self.batch_size = batch_size
        assert alpha > 0, "The prioritization value 'alpha' must be positive!"
        self.alpha = alpha
        self.beta = beta
        self.storage = deque(maxlen=storage_size)
        self.priorities = np.zeros((storage_size,), dtype=np.float32)
        self.position = 0

    def __len__(self) -> int:
        """Return the number of transitions in storage."""
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
        observations: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        infos: Dict[str, Any],
        next_observations: th.Tensor,
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
        max_prio = self.priorities.max() if self.storage else 1.0
        self.priorities[self.position] = max_prio
        self.storage.append(transition)
        self.position = (self.position + 1) % self.storage_size

    def sample(self, step: int) -> PrioritizedReplayBatch:
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
        
        # compute probabilities and sample indices
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.storage), self.batch_size, p=probs)

        # get samples
        samples = [self.storage[i] for i in indices]
        weights = (len(self.storage) * probs[indices]) ** (-self.annealing_beta(step))
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

        return PrioritizedReplayBatch(observations=self.to_torch(obs),
                                      actions=self.to_torch(actions),
                                      rewards=self.to_torch(rewards),
                                      terminateds=self.to_torch(terminateds),
                                      truncateds=self.to_torch(truncateds),
                                      next_observations=self.to_torch(next_obs),
                                      weights=self.to_torch(weights),
                                      indices=indices)

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
