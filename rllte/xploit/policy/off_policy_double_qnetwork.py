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
from typing import Any, Dict, Optional, Tuple, Type

import gymnasium as gym
import torch as th
from torch import nn
from torch.distributions import Distribution

from rllte.common.base_policy import BasePolicy
from rllte.common.utils import ExportModel


class DoubleCritic(nn.Module):
    """Double critic network for DrQv2 and SAC.

    Args:
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Critic network instance.
    """

    def __init__(self, action_dim: int, feature_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, ...]:
        """Value estimation.

        Args:
            obs (th.Tensor): Observations.
            action (th.Tensor): Actions.

        Returns:
            Estimated values.
        """
        h_action = th.cat([obs, action], dim=-1)

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class OffPolicyDoubleQNetwork(BasePolicy):
    """Q-network for off-policy algortithms like `DQN`.

        Structure: self.encoder (shared by actor and critic), self.qnet, self.qnet_target
        Optimizers: self.opt -> (self.qnet, self.qnet_target)

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        opt_class (Type[th.optim.Optimizer]): Optimizer class.
        opt_kwargs (Optional[Dict[str, Any]]): Optimizer keyword arguments.
        init_fn (Optional[str]): Parameters initialization method.

    Returns:
        Actor network instance.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 64,
        hidden_dim: int = 1024,
        opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
        opt_kwargs: Optional[Dict[str, Any]] = None,
        init_fn: Optional[str] = None,
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

        # build q-network and target q-network
        self.qnet = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.qnet_target = deepcopy(self.qnet)

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
        self.encoder = encoder
        # initialize parameters
        self.apply(self.init_fn)
        # synchronize the parameters of Q-network and target Q-network
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        # build optimizers
        self.opt = self.opt_class(self.parameters(), **self.opt_kwargs)

    def explore(self, obs: th.Tensor) -> th.Tensor:
        """Explore the environment and randomly generate actions.

        Args:
            obs (th.Tensor): Observation from the environment.

        Returns:
            Sampled actions.
        """
        return th.randint(low=self.action_range[0], high=self.action_range[1], size=(obs.size()[0],), device=obs.device)

    def forward(self, obs: th.Tensor, training: bool = True, step: int = 0) -> th.Tensor:
        """Sample actions based on observations.

        Args:
            obs (th.Tensor): Observations.
            training (bool): Training mode, True or False.
            step (int): Global training step.

        Returns:
            Sampled actions.
        """
        encoded_obs = self.encoder(obs)
        actions = self.qnet(encoded_obs).argmax(dim=1).reshape(-1)

        return actions

    def save(self, path: Path, pretraining: bool = False) -> None:
        """Save models.

        Args:
            path (Path): Save path.
            pretraining (bool): Pre-training mode.

        Returns:
            None.
        """
        if pretraining:  # pretraining
            th.save(self.state_dict(), path / "pretrained.pth")
        else:
            export_model = ExportModel(encoder=self.encoder, actor=self.qnet)
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
        self.load_state_dict(params)
