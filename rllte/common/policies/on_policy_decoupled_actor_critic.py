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


import os
from pathlib import Path
from typing import Tuple

import torch as th
from torch import nn
from torch.nn import functional as F

from rllte.common.policies.on_policy_shared_actor_critic import BoxActor, DiscreteActor, MultiBinaryActor
from rllte.common.utils import ExportModel


class OnPolicyDecoupledActorCritic(nn.Module):
    """Actor-Critic network using using separate encoders for on-policy algorithms like `DAAC`.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        action_type (str): The action type like 'Discrete' or 'Box', etc.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor-Critic network instance.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        action_type: str,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_type = action_type
        if action_type == "Discrete":
            self.actor = DiscreteActor(
                obs_shape=obs_shape, action_dim=action_dim, feature_dim=feature_dim, hidden_dim=hidden_dim
            )
        elif action_type == "Box":
            self.actor = BoxActor(obs_shape=obs_shape, action_dim=action_dim, feature_dim=feature_dim, hidden_dim=hidden_dim)

        elif action_type == "MultiBinary":
            self.actor = MultiBinaryActor(
                obs_shape=obs_shape, action_dim=action_dim, feature_dim=feature_dim, hidden_dim=hidden_dim
            )
        else:
            raise NotImplementedError("Unsupported action type!")

        if len(obs_shape) > 1:
            self.gae = nn.Linear(feature_dim + action_dim, 1)
            self.critic = nn.Linear(feature_dim, 1)
        else:
            # for state-based observations and `IdentityEncoder`
            self.gae = nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.critic = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        # placeholder for distribution
        self.actor_encoder = None
        self.critic_encoder = None
        self.dist = None

    def get_action_and_value(self, obs: th.Tensor, training: bool = True) -> th.Tensor:
        """Get actions and estimated values for observations.

        Args:
            obs (th.Tensor): Observations.
            training (bool): training mode, `True` or `False`.

        Returns:
            Sampled actions, estimated values, and log of probabilities for observations when `training` is `True`,
            else only deterministic actions.
        """
        h = self.actor_encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        dist = self.dist(*policy_outputs)

        if training:
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            return actions, self.critic(self.critic_encoder(obs)), log_probs
        else:
            actions = dist.mean
            return actions

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values for observations.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Estimated values.
        """
        return self.critic(self.critic_encoder(obs))

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor = None) -> Tuple[th.Tensor, ...]:
        """Evaluate actions according to the current policy given the observations.

        Args:
            obs (th.Tensor): Sampled observations.
            actions (th.Tensor): Sampled actions.

        Returns:
            Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        h = self.actor_encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        dist = self.dist(*policy_outputs)

        if self.action_type == "Discrete":
            encoded_actions = F.one_hot(actions.long(), self.action_dim).to(h.device)
        else:
            encoded_actions = actions

        log_probs = dist.log_prob(actions)
        gae = self.gae(th.cat([h, encoded_actions], dim=1))
        entropy = dist.entropy().mean()

        return gae, self.critic(self.critic_encoder(obs)), log_probs, entropy

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
            export_model = ExportModel(encoder=self.actor_encoder, actor=self.actor)
            th.save(export_model, path / "agent.pth")

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        params = th.load(os.path.join(path, "pretrained.pth"), map_location=self.device)
        self.load_state_dict(params)