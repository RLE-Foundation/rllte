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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
from torch import nn

from rllte.common.initialization import get_init_fn
from rllte.common.preprocessing import process_observation_space, process_action_space


class BasePolicy(ABC, nn.Module):
    """Base class for all policies.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        opt_class (Type[th.optim.Optimizer]): Optimizer class.
        opt_kwargs (Optional[Dict[str, Any]]): Optimizer keyword arguments.
        init_fn (str): Parameters initialization method.

    Returns:
        Base policy instance.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int,
        hidden_dim: int,
        opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
        opt_kwargs: Optional[Dict[str, Any]] = None,
        init_fn: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        assert feature_dim > 0, "The `feature_dim` should be positive!"
        assert hidden_dim > 0, "The `hidden_dim` should be positive!"
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.opt_class = opt_class
        self.opt_kwargs = opt_kwargs
        self.init_fn = get_init_fn(init_fn)

        # get environment information
        self.obs_shape = process_observation_space(observation_space)
        self.action_shape, self.action_dim, self.policy_action_dim, self.action_type = process_action_space(action_space)
        self.nvec = tuple(int(_) for _ in action_space.nvec.reshape(-1)) if hasattr(action_space, "nvec") else None

        # placeholder for optimizers
        self._optimizers: Dict[str, th.optim.Optimizer] = {}
        
    @property
    def optimizers(self) -> Dict[str, th.optim.Optimizer]:
        """Get optimizers."""
        return self._optimizers

    @staticmethod
    @abstractmethod
    def describe() -> None:
        """Describe the policy."""

    def explore(self, obs: th.Tensor) -> th.Tensor:
        """Explore the environment and randomly generate actions.

        Args:
            obs (th.Tensor): Observation from the environment.

        Returns:
            Sampled actions.
        """

    @abstractmethod
    def forward(self, obs: th.Tensor, training: bool = True) -> Union[th.Tensor, Tuple[th.Tensor]]:
        """Forward method.

        Args:
            obs (th.Tensor): Observation from the environment.
            training (bool): Whether the agent is being trained or not.

        Returns:
            Sampled actions, estimated values, ..., depends on specific algorithms.
        """

    @abstractmethod
    def freeze(self) -> None:
        """Freeze the policy and start training."""

    @abstractmethod
    def save(self, path: Path, pretraining: bool = False) -> None:
        """Save models.

        Args:
            path (Path): Save path.
            pretraining (bool): Pre-training mode.

        Returns:
            None.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
