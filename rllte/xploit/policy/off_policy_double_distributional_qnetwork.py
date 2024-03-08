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
from typing import Any, Dict, Optional, Type

import gymnasium as gym
import torch as th
from torch import nn
from torch.distributions import Distribution

from rllte.common.prototype import BasePolicy
from rllte.common.utils import ExportModel


class OffPolicyDoubleDistributionalQNetwork(BasePolicy):
    """Distributional Q-network for off-policy algortithms like `C51` or `Rainbow`.

        Structure: self.encoder (shared by actor and critic), self.qnet, self.qnet_target
        Optimizers: self.opt -> (self.qnet, self.qnet_target)

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        opt_class (Type[th.optim.Optimizer]): Optimizer class.
        opt_kwargs (Dict[str, Any]): Optimizer keyword arguments.
        init_fn (str): Parameters initialization method.

    Returns:
        Actor network instance.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        n_atoms: int = 101,
        v_min: float = -100.0,
        v_max: float = 100,
        opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
        opt_kwargs: Optional[Dict[str, Any]] = None,
        init_fn: str = "orthogonal",
    ) -> None:
        if opt_kwargs is None:
            opt_kwargs = {}
        assert isinstance(action_space, gym.spaces.Discrete), "Only discrete action space is supported!"
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=opt_class,
            opt_kwargs=opt_kwargs,
            init_fn=init_fn,
        )
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer("atoms", th.linspace(v_min, v_max, steps=n_atoms))
        self.register_buffer("target_atoms", th.linspace(v_min, v_max, steps=n_atoms))

        # build q-network and target q-network
        self.qnet = nn.Sequential(
            nn.Linear(self.feature_dim, self.policy_action_dim * self.n_atoms),
        )
        self.qnet_target = deepcopy(self.qnet)

    @staticmethod
    def describe() -> None:
        """Describe the policy."""
        print("\n")
        print("=" * 80)
        print(f"{'Name'.ljust(10)} : OffPolicyDoubleQNetwork")
        print(f"{'Structure'.ljust(10)} : self.encoder (shared by actor and critic), self.qnet, self.qnet_target")
        print(f"{'Forward'.ljust(10)} : obs -> self.encoder -> self.qnet -> action values")
        print(f"{'Optimizers'.ljust(10)} : self.optimizers['opt'] -> self.qnet")
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
        self.encoder = encoder
        # initialize parameters
        self.apply(self.init_fn)
        # synchronize the parameters of Q-network and target Q-network
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        # build optimizers
        self._optimizers["opt"] = self.opt_class(self.parameters(), **self.opt_kwargs)

    def forward(self, obs: th.Tensor, training: bool = True) -> th.Tensor:
        """Sample actions based on observations.

        Args:
            obs (th.Tensor): Observations.
            training (bool): Training mode, True or False.

        Returns:
            Sampled actions.
        """
        encoded_obs = self.encoder(obs)
        logits = self.qnet(encoded_obs)
        pmfs = th.softmax(logits.view(len(obs), self.policy_action_dim, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        actions = th.argmax(q_values, 1)
        return actions
    
    def get_online_dist(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        encoded_obs = self.encoder(obs)
        logits = self.qnet(encoded_obs)
        pmfs = th.softmax(logits.view(len(obs), self.policy_action_dim, self.n_atoms), dim=2)
        dist = pmfs[th.arange(len(obs)), action.long()]
        return dist

    def get_target_dist(self, obs: th.Tensor) -> th.Tensor:
        encoded_obs = self.encoder(obs)
        logits = self.qnet_target(encoded_obs)
        pmfs = th.softmax(logits.view(len(obs), self.policy_action_dim, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        action = th.argmax(q_values, 1)
        dist = pmfs[th.arange(len(obs)), action]
        return dist

    def save(self, path: Path, pretraining: bool, global_step: int) -> None:
        """Save models.

        Args:
            path (Path): Save path.
            pretraining (bool): Pre-training mode.
            global_step (int): Global training step.

        Returns:
            None.
        """
        if pretraining:  # pretraining
            th.save(self.state_dict(), path / f"pretrained_{global_step}.pth")
        else:
            export_model = ExportModel(encoder=self.encoder, actor=self.qnet)
            th.save(export_model, path / f"agent_{global_step}.pth")
