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
from typing import Any, Dict, Optional, Tuple, Type

import gymnasium as gym
import torch as th
from torch import nn
from torch.distributions import Distribution

from rllte.common.prototype import BasePolicy
from rllte.common.utils import ExportModel

from .utils import OffPolicyDoubleCritic


class OffPolicyDetActorDoubleCritic(BasePolicy):
    """Deterministic actor network and double critic network for off-policy algortithms like `DrQv2`, `DDPG`.
        Here the 'self.dist' refers to an action noise instance.

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

        # build actor and critic
        self.actor = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Tanh(),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.policy_action_dim),
            nn.Tanh(),
        )

        self.critic = OffPolicyDoubleCritic(action_dim=self.policy_action_dim, feature_dim=self.feature_dim, hidden_dim=hidden_dim)
        self.critic_target = OffPolicyDoubleCritic(action_dim=self.policy_action_dim, feature_dim=self.feature_dim, hidden_dim=self.hidden_dim)

    def describe() -> None:
        """Describe the policy."""
        print("\n")
        print("=" * 80)
        print(f"{'Name'.ljust(10)} : OffPolicyDetActorDoubleCritic")
        print(f"{'Structure'.ljust(10)} : self.encoder (shared by actor and critic), self.actor")
        print(f"{''.ljust(10)} : self.critic, self.critic_target")
        print(f"{'Forward'.ljust(10)} : obs -> self.encoder -> self.actor -> actions")
        print(f"{''.ljust(10)} : obs -> self.encoder -> self.critic -> double values")
        print(f"{'Optimizers'.ljust(10)} : self.optimizers['encoder_opt'] -> self.encoder")
        print(f"{''.ljust(10)} : self.optimizers['critic_opt'] -> (self.encoder, self.critic)")
        print(f"{''.ljust(10)} : self.optimizers['actor_opt'] -> self.actor")
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
        # set distribution
        assert dist is not None, "Distribution should not be None!"
        self.dist = dist
        # initialize parameters
        self.apply(self.init_fn)
        # synchronize the parameters of critic and target critic
        self.critic_target.load_state_dict(self.critic.state_dict())
        # build optimizers
        self._optimizers['encoder_opt'] = self.opt_class(self.encoder.parameters(), **self.opt_kwargs)
        self._optimizers['actor_opt'] = self.opt_class(self.actor.parameters(), **self.opt_kwargs)
        self._optimizers['critic_opt'] = self.opt_class(self.critic.parameters(), **self.opt_kwargs)

    def explore(self, obs: th.Tensor) -> th.Tensor:
        """Explore the environment and randomly generate actions.

        Args:
            obs (th.Tensor): Observation from the environment.

        Returns:
            Sampled actions.
        """
        return th.rand(size=(obs.size()[0], self.policy_action_dim), device=obs.device).uniform_(-1.0, 1.0)

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
        dist = self.get_dist(obs=encoded_obs, step=step)

        if not training:
            actions = dist.mean
        else:
            actions = dist.sample()

        return actions

    def get_dist(self, obs: th.Tensor, step: int) -> Distribution:
        """Get sample distribution.

        Args:
            obs (th.Tensor): Observations.
            step (int): Global training step.

        Returns:
            RLLTE distribution.
        """
        mu = self.actor(obs)

        # for Scheduled Exploration Noise
        self.dist.reset(mu, step)

        return self.dist

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
            export_model = ExportModel(encoder=self.encoder, actor=self.actor)
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
