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


from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
from torch import nn


class BasePolicy(nn.Module):
    """Base class for all policies.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        opt_class (Type[th.optim.Optimizer]): Optimizer class.
        opt_kwargs (Optional[Dict[str, Any]]): Optimizer keyword arguments.
        init_method (Callable): Initialization method.

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
        init_method: Callable = nn.init.orthogonal_,
    ) -> None:
        super().__init__()

        assert feature_dim > 0, "The `feature_dim` should be positive!"
        assert hidden_dim > 0, "The `hidden_dim` should be positive!"
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.opt_class = opt_class
        self.opt_kwargs = opt_kwargs
        self.init_method = init_method

        # get environment information
        self.obs_shape = observation_space.shape
        if action_space.__class__.__name__ == "Discrete":
            self.action_shape = action_space.shape
            self.action_dim = int(action_space.n)
            self.action_type = "Discrete"
            self.action_range = [0, int(action_space.n) - 1]
        elif action_space.__class__.__name__ == "Box":
            self.action_shape = action_space.shape
            self.action_dim = action_space.shape[0]
            self.action_type = "Box"
            self.action_range = [
                float(action_space.low[0]),
                float(action_space.high[0]),
            ]
        elif action_space.__class__.__name__ == "MultiBinary":
            self.action_shape = action_space.shape
            self.action_dim = action_space.shape[0]
            self.action_type = "MultiBinary"
            self.action_range = [0, 1]
        else:
            raise NotImplementedError("Unsupported action type!")

    def act(self, obs: th.Tensor, training: bool = True) -> Union[th.Tensor, Tuple[th.Tensor]]:
        """Select an action from the input observation.

        Args:
            obs (th.Tensor): Observation from the environment.
            training (bool): Whether the agent is being trained or not.

        Returns:
            Sampled actions, estimated values, ..., depends on specific algorithms.
        """

    def freeze(self) -> None:
        """Freeze the policy."""

    def save(self, path: Path, pretraining: bool = False) -> None:
        """Save models.

        Args:
            path (Path): Save path.
            pretraining (bool): Pre-training mode.

        Returns:
            None.
        """

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
