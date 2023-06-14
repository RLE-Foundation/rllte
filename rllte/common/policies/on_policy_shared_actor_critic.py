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
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import torch as th
from torch import nn

from rllte.common.utils import ExportModel


class DiscreteActor(nn.Module):
    """Actor for `Discrete` tasks.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if len(obs_shape) > 1:
            self.actor = nn.Linear(feature_dim, action_dim)
        else:
            # for state-based observations and `IdentityEncoder`
            self.actor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )

    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor]:
        """Get policy outputs for training.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Unnormalized probabilities.
        """
        logits = self.actor(obs)
        return (logits,)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Unnormalized action probabilities.
        """
        return self.actor(obs)


class BoxActor(nn.Module):
    """Actor for `Box` tasks.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if len(obs_shape) > 1:
            self.actor_mu = nn.Linear(feature_dim, action_dim)
        else:
            # for state-based observations and `IdentityEncoder`
            self.actor_mu = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )
        self.actor_logstd = nn.Parameter(th.ones(1, action_dim))

    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Get policy outputs for training.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Mean and variance of sample distributions.
        """
        mu = self.actor_mu(obs)
        logstd = self.actor_logstd.expand_as(mu)
        return (mu, logstd.exp())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        return self.actor_mu(obs)


class MultiBinaryActor(nn.Module):
    """Actor for `MultiBinary` tasks.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if len(obs_shape) > 1:
            self.actor = nn.Linear(feature_dim, action_dim)
        else:
            # for state-based observations and `IdentityEncoder`
            self.actor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )

    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor]:
        """Get policy outputs for training.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Unnormalized probabilities.
        """
        logits = self.actor(obs)
        return (logits,)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Unnormalized action probabilities.
        """
        return self.actor(obs)


class OnPolicySharedActorCritic(nn.Module):
    """Actor-Critic network using a shared encoder for on-policy algorithms like `PPO`.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        action_type (str): The action type like 'Discrete' or 'Box', etc.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        aux_critic (bool): Use auxiliary critic or not, for `PPG` agent.

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
        aux_critic: bool = False,
    ) -> None:
        super().__init__()
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
            self.critic = nn.Linear(feature_dim, 1)
        else:
            # for state-based observations and `IdentityEncoder`
            self.critic = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        if aux_critic:
            self.aux_critic = deepcopy(self.critic)

        # placeholder for distribution
        self.encoder = None
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
        h = self.encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        dist = self.dist(*policy_outputs)

        if training:
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            return actions, self.critic(h), log_probs
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
        return self.critic(self.encoder(obs))

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor = None) -> Tuple[th.Tensor, ...]:
        """Evaluate actions according to the current policy given the observations.

        Args:
            obs (th.Tensor): Sampled observations.
            actions (th.Tensor): Sampled actions.

        Returns:
            Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        h = self.encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        dist = self.dist(*policy_outputs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return self.critic(h), log_probs, entropy

    def get_dist_and_aux_value(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        """Get probs and auxiliary estimated values for auxiliary phase update.

        Args:
            obs: Sampled observations.

        Returns:
            Sample distribution, estimated values, auxiliary estimated values.
        """
        h = self.encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        dist = self.dist(*policy_outputs)

        return dist, self.critic(h.detach()), self.aux_critic(h)

    def get_policy_outputs(self, obs: th.Tensor) -> th.Tensor:
        """Get policy outputs for training.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Policy outputs like unnormalized probabilities for `Discrete` tasks.
        """
        h = self.encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        return th.cat(policy_outputs, dim=1)

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

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        params = th.load(os.path.join(path, "pretrained.pth"), map_location=self.device)
        self.load_state_dict(params)
