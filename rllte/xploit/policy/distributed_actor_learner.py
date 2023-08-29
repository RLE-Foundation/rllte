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


from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
from torch import nn
from torch.distributions import Distribution
from torch.nn import functional as F

from rllte.common.prototype import BasePolicy
from rllte.common.utils import ExportModel
from .utils import DisctributedActorCritic


class DistributedActorLearner(BasePolicy):
    """Actor-Learner network for IMPALA.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        opt_class (Type[th.optim.Optimizer]): Optimizer class.
        opt_kwargs (Optional[Dict[str, Any]]): Optimizer keyword arguments.
        init_fn (Optional[str]): Parameters initialization method.
        use_lstm (bool): Whether to use LSTM module.

    Returns:
        Actor-Critic network.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int,
        hidden_dim: int = 512,
        opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
        opt_kwargs: Optional[Dict[str, Any]] = None,
        init_fn: Optional[str] = None,
        use_lstm: bool = False
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=opt_class,
            opt_kwargs=opt_kwargs,
            init_fn=init_fn,
        )

        # TODO: add support for LSTM
        self.actor = DisctributedActorCritic(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            action_dim=self.policy_action_dim,
            action_type=self.action_type,
            feature_dim=self.feature_dim
        )
        self.learner = DisctributedActorCritic(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            action_dim=self.policy_action_dim,
            action_type=self.action_type,
            feature_dim=self.feature_dim
        )
    
    def describe() -> None:
        """Describe the policy."""
        print("\n")
        print("=" * 80)
        print(f"{'Name'.ljust(10)} : DistributedActorLearner")
        print(f"{'Structure'.ljust(10)} : self.actor, self.learner")
        print(f"{'Forward'.ljust(10)} : obs, last actions, rewards -> self.actor -> actions, values, policy outputs")
        print(f"{''.ljust(10)} : obs, last actions, rewards -> self.learner -> actions, values, policy outputs")
        print(f"{'Optimizers'.ljust(10)} : self.optimizers['opt'] -> self.learner")
        print("=" * 80)
        print("\n")

    def freeze(self, encoder: nn.Module, dist: Distribution) -> None:
        """Freeze all the elements like `encoder` and `dist`.

        Args:
            encoder (nn.Module): Encoder network.
            dist (Distribution): Distribution class.

        Returns:
            None.
        """
        # set encoder
        assert encoder is not None, "Encoder should not be None!"
        self.actor.encoder = encoder
        self.learner.encoder = deepcopy(encoder)
        # set distribution
        assert dist is not None, "Distribution should not be None!"
        self.actor.dist = dist
        self.learner.dist = dist
        # initialize parameters
        self.actor.apply(self.init_fn)
        self.learner.apply(self.init_fn)
        # synchronize the parameters of actor and learner
        self.actor.load_state_dict(self.learner.state_dict())
        # share memory
        self.actor.share_memory()
        # build optimizers
        self._optimizers['opt'] = self.opt_class(self.learner.parameters(), **self.opt_kwargs)
    
    def forward(self, *args) -> th.Tensor:
        """Only for inference."""

    def to(self, device: th.device) -> None:
        """Only move the learner to device, and keep actor in CPU.

        Args:
            device (th.device): Device to use.

        Returns:
            None.
        """
        self.learner.to(device)

    def save(self, path: Path) -> None:
        """Save models.

        Args:
            path (Path): Save path.

        Returns:
            None.
        """
        export_model = ExportModel(encoder=self.learner.encoder, actor=self.learner.actor)
        th.save(export_model, path / "agent.pth")

    def load(self, path: str, device: th.device) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.
            device (th.device): Device to use.

        Returns:
            None.
        """
        params = th.load(path, map_location=device)
        self.actor.load_state_dict(params)
        self.learner.load_state_dict(params)
