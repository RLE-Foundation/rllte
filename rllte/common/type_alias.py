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


from abc import abstractmethod
from typing import Dict, Generator, Iterable, NamedTuple, Tuple, Union

import numpy as np
import torch as th

from rllte.common.preprocessing import ObsShape as ObsShape
from rllte.common.prototype.base_distribution import BaseDistribution
from rllte.common.prototype.base_policy import BasePolicy
from rllte.common.prototype.base_storage import BaseStorage
from rllte.env.utils import Gymnasium2Torch, GymObs

VecEnv = Gymnasium2Torch


class RolloutStorageType(BaseStorage):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # data containers
        self.observations: Union[th.Tensor, Dict[str, th.Tensor]]
        self.actions: th.Tensor
        self.rewards: th.Tensor
        self.terminateds: th.Tensor
        self.truncateds: th.Tensor
        self.log_probs: th.Tensor
        self.values: th.Tensor
        self.returns: th.Tensor
        self.advantages: th.Tensor

    def add(self, *args, **kwargs) -> None:
        """Add a transition to the storage."""

    def update(self, *args, **kwargs) -> None:
        """Update the storage."""

    def compute_returns_and_advantages(self, last_values: th.Tensor) -> None:
        """Generalized advantage estimation (GAE)."""

    def sample(self) -> Generator:
        """Sample from the storage."""
        yield


class OnPolicyType(BasePolicy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # for decoupled actor-critic
        self.actor_opt: th.optim.Optimizer
        self.critic_opt: th.optim.Optimizer
        self.actor_params: Iterable[th.Tensor]
        self.critic_params: Iterable[th.Tensor]

    @abstractmethod
    def get_value(self, obs: GymObs) -> th.Tensor:
        """Get the value of the state."""

    @abstractmethod
    def evaluate_actions(self, obs: GymObs, actions: th.Tensor) -> th.Tensor:
        """Evaluate the action."""

    @abstractmethod
    def get_policy_outputs(self, obs: th.Tensor) -> th.Tensor:
        """Get policy outputs for training."""

    @abstractmethod
    def get_dist_and_aux_value(self, obs: th.Tensor) -> Tuple[BaseDistribution, th.Tensor, th.Tensor]:
        """Get probs and auxiliary estimated values for auxiliary phase update."""


class OffPolicyType(BasePolicy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.actor: th.nn.Module
        self.critic: th.nn.Module
        self.actor_target: th.nn.Module
        self.critic_target: th.nn.Module
        self.qnet: th.nn.Module
        self.qnet_target: th.nn.Module

    @abstractmethod
    def get_dist(self, *args, **kwargs) -> BaseDistribution:
        """Get the sampling distribution."""


class ReplayStorageType(BaseStorage):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class DistributedPolicyType(BasePolicy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.actor: th.nn.Module
        self.learner: th.nn.Module


class DistributedStorageType(BaseStorage):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


# batch types
class VanillaRolloutBatch(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    values: th.Tensor
    returns: th.Tensor
    terminateds: th.Tensor
    truncateds: th.Tensor
    old_log_probs: th.Tensor
    adv_targ: th.Tensor


class DictRolloutBatch(NamedTuple):
    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    values: th.Tensor
    returns: th.Tensor
    terminateds: th.Tensor
    truncateds: th.Tensor
    old_log_probs: th.Tensor
    adv_targ: th.Tensor


class VanillaReplayBatch(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    terminateds: th.Tensor
    truncateds: th.Tensor
    next_observations: th.Tensor


class DictReplayBatch(NamedTuple):
    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    rewards: th.Tensor
    terminateds: th.Tensor
    truncateds: th.Tensor
    next_observations: Dict[str, th.Tensor]


class PrioritizedReplayBatch(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    terminateds: th.Tensor
    truncateds: th.Tensor
    next_observations: th.Tensor
    indices: np.ndarray
    weights: th.Tensor


class NStepReplayBatch(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor
    next_observations: th.Tensor
