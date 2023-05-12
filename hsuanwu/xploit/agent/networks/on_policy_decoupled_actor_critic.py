from typing import Tuple

import torch as th
from torch import nn
from torch.nn import functional as F


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
            obs (Tensor): Observations.

        Returns:
            Unnormalized probabilities.
        """
        logits = self.actor(obs)
        return (logits,)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (Tensor): Observations.

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
            obs (Tensor): Observations.

        Returns:
            Mean and variance of sample distributions.
        """
        mu = self.actor_mu(obs)
        logstd = self.actor_logstd.expand_as(mu)
        return (mu, logstd.exp())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (Tensor): Observations.

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
            obs (Tensor): Observations.

        Returns:
            Unnormalized probabilities.
        """
        logits = self.actor(obs)
        return (logits,)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (Tensor): Observations.

        Returns:
            Unnormalized action probabilities.
        """
        return self.actor(obs)


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

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        return self.actor(self.actor_encoder(obs))

    def get_action_and_value(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        """Get actions and estimated values for observations.

        Args:
            obs (Tensor): Sampled observations.

        Returns:
            Sampled actions, Estimated values, log of the probability evaluated at `actions`.
        """
        h = self.actor_encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        dist = self.dist(*policy_outputs)

        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        if self.action_type == "Discrete":
            encoded_actions = F.one_hot(actions.long(), self.action_dim).to(h.device)
        else:
            encoded_actions = actions
        gae = self.gae(th.cat([h, encoded_actions], dim=1))

        return actions, gae, self.critic(self.critic_encoder(obs)), log_probs

    def get_det_action(self, obs: th.Tensor) -> th.Tensor:
        """Get deterministic actions for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        policy_outputs = self.actor.get_policy_outputs(self.actor_encoder(obs))
        dist = self.dist(*policy_outputs)

        return dist.mean

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        return self.critic(self.critic_encoder(obs))

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor = None) -> Tuple[th.Tensor, ...]:
        """Get actions and estimated values for observations.

        Args:
            obs (Tensor): Sampled observations.
            actions (Tensor): Sampled actions.

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
