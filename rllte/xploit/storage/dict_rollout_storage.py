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


from typing import Dict, Generator

import gymnasium as gym
import torch as th
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from rllte.common.type_alias import VanillaRolloutBatch
from rllte.xploit.storage.vanilla_rollout_storage import VanillaRolloutStorage


class DictRolloutStorage(VanillaRolloutStorage):
    """Dict Rollout storage for on-policy algorithms and dictionary observations.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        device (str): Device to convert the data.
        storage_size (int): The capacity of the storage. Here it refers to the length of per rollout.
        batch_size (int): Batch size of samples.
        num_envs (int): The number of parallel environments.
        discount (float): The discount factor.
        gae_lambda (float): Weighting coefficient for generalized advantage estimation (GAE).

    Returns:
        Dict rollout storage.
    """

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 device: str = "cpu",
                 storage_size: int = 256,
                 batch_size: int = 64,
                 num_envs: int = 8,
                 discount: float = 0.999,
                 gae_lambda: float = 0.95,
                 ) -> None:
        super(VanillaRolloutStorage, self).__init__(observation_space, action_space, device, storage_size, batch_size, num_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutStorage only support Dict observation space."

        self.discount = discount
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        """Reset the storage."""
        # data containers
        self.observations = dict()
        for key, shape in self.obs_shape.items():
            self.observations[key] = th.empty(size=(self.storage_size + 1, self.num_envs, *shape), dtype=th.float32, device=self.device)
        self.actions = th.empty(size=(self.storage_size, self.num_envs, self.action_dim), dtype=th.float32, device=self.device)
        self.rewards = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.terminateds = th.empty(size=(self.storage_size + 1, self.num_envs), dtype=th.float32, device=self.device)
        self.truncateds = th.empty(size=(self.storage_size + 1, self.num_envs), dtype=th.float32, device=self.device)
        # first next_terminated
        self.terminateds[0].fill_(0.0)
        self.truncateds[0].fill_(0.0)
        # extra part
        self.log_probs = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.values = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.returns = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.advantages = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)

        super(VanillaRolloutStorage, self).reset()

    def add(self,
            observations: Dict[str, th.Tensor],
            actions: th.Tensor,
            rewards: th.Tensor,
            terminateds: th.Tensor,
            truncateds: th.Tensor,
            infos: Dict,
            next_observations: th.Tensor,
            log_probs: th.Tensor,
            values: th.Tensor
            ) -> None:
        """Add sampled transitions into storage.

        Args:
            observations (Dict[str, th.Tensor]): Observations.
            actions (th.Tensor): Actions.
            rewards (th.Tensor): Rewards.
            terminateds (th.Tensor): Termination signals.
            truncateds (th.Tensor): Truncation signals.
            infos (Dict): Extra information.
            next_observations (th.Tensor): Next observations.
            log_probs (th.Tensor): Log of the probability evaluated at `actions`.
            values (th.Tensor): Estimated values.

        Returns:
            None.
        """
        for key in self.observations.keys():
            if isinstance(self.observation_space.spaces[key], gym.spaces.Discrete):
                obs_ = observations[key].reshape((self.num_envs,) + self.obs_shape[key])
                next_obs_ = next_observations[key].reshape((self.num_envs,) + self.obs_shape[key])
            else:
                obs_ = observations[key]
                next_obs_ = next_observations[key]

            self.observations[key][self.step].copy_(obs_)
            self.observations[key][self.step + 1].copy_(next_obs_)

        self.actions[self.step].copy_(actions.view(self.num_envs, self.action_dim))
        self.rewards[self.step].copy_(rewards)
        self.terminateds[self.step + 1].copy_(terminateds)
        self.truncateds[self.step + 1].copy_(truncateds)
        self.log_probs[self.step].copy_(log_probs)
        self.values[self.step].copy_(values.flatten())

        self.full = True if self.step == self.storage_size - 1 else False
        self.step = (self.step + 1) % self.storage_size

    def sample(self) -> Generator:
        """Sample data from storage."""
        assert self.full, "Cannot sample when the storage is not full!"
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_envs * self.num_steps)), self.batch_size, drop_last=True)

        for indices in sampler:
            batch_obs = {key: item[:-1].view(-1, *self.obs_shape[key])[indices] for (key, item) in self.observations.items()}
            batch_actions = self.actions.view(-1, *self.action_shape)[indices]
            batch_values = self.values.view(-1)[indices]
            batch_returns = self.returns.view(-1)[indices]
            batch_terminateds = self.terminateds[:-1].view(-1)[indices]
            batch_truncateds = self.truncateds[:-1].view(-1)[indices]
            batch_old_log_probs = self.log_probs.view(-1)[indices]
            adv_targ = self.advantages.view(-1)[indices]

            yield VanillaRolloutBatch(
                observations=batch_obs,
                actions=batch_actions,
                values=batch_values,
                returns=batch_returns,
                terminateds=batch_terminateds,
                truncateds=batch_truncateds,
                old_log_probs=batch_old_log_probs,
                adv_targ=adv_targ
            )
