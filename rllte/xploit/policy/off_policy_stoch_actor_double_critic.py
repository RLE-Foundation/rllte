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

from rllte.common.prototype import BaseDistribution as Distribution
from rllte.common.prototype import BasePolicy
from rllte.common.utils import ExportModel

from .utils import OffPolicyDoubleCritic, get_off_policy_actor

# from torch.distributions import Distribution


class OffPolicyStochActorDoubleCritic(BasePolicy):
    """Stochastic actor network and double critic network for off-policy algortithms like `SAC`.
        Here the 'self.dist' refers to an sampling distribution instance.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        opt_class (Type[th.optim.Optimizer]): Optimizer class.
        opt_kwargs (Dict[str, Any]): Optimizer keyword arguments.
        log_std_range (Tuple): Range of log standard deviation.
        init_fn (str): Parameters initialization method.

    Returns:
        Actor-Critic network.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 64,
        hidden_dim: int = 1024,
        opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
        opt_kwargs: Optional[Dict[str, Any]] = None,
        log_std_range: Tuple = (-5, 2),
        init_fn: str = "orthogonal",
    ) -> None:
        if opt_kwargs is None:
            opt_kwargs = {}
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
        actor_kwargs = {"action_dim": self.policy_action_dim, "hidden_dim": self.hidden_dim, "feature_dim": self.feature_dim}
        if self.action_type == "Box":
            actor_kwargs["log_std_range"] = log_std_range # type: ignore[assignment]

        self.actor = get_off_policy_actor(action_type=self.action_type, actor_kwargs=actor_kwargs)

        self.critic = OffPolicyDoubleCritic(
            action_dim=self.policy_action_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            action_type=self.action_type,
        )
        self.critic_target = OffPolicyDoubleCritic(
            action_dim=self.policy_action_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            action_type=self.action_type,
        )

    @staticmethod
    def describe() -> None:
        """Describe the policy."""
        print("\n")
        print("=" * 80)
        print(f"{'Name'.ljust(10)} : OffPolicyStochActorDoubleCritic")
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
        self._optimizers["encoder_opt"] = self.opt_class(self.encoder.parameters(), **self.opt_kwargs)
        self._optimizers["actor_opt"] = self.opt_class(self.actor.parameters(), **self.opt_kwargs)
        self._optimizers["critic_opt"] = self.opt_class(self.critic.parameters(), **self.opt_kwargs)

    def forward(self, obs: th.Tensor, training: bool = True) -> th.Tensor:
        """Sample actions based on observations.

        Args:
            obs (th.Tensor): Observations.
            training (bool): Training mode, True or False.

        Returns:
            Sampled actions.
        """
        encoded_obs = self.encoder(obs)
        dist = self.get_dist(obs=encoded_obs)
        if not training:
            actions = dist.mean
        else:
            actions = dist.sample()

        return actions

    def get_dist(self, obs: th.Tensor) -> Distribution:
        """Get sample distribution.

        Args:
            obs (th.Tensor): Observations.
            step (int): Global training step.

        Returns:
            Action distribution.
        """
        policy_outputs = self.actor.get_policy_outputs(obs)

        return self.dist(*policy_outputs)

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
            export_model = ExportModel(encoder=self.encoder, actor=self.actor)
            th.save(export_model, path / f"agent_{global_step}.pth")
