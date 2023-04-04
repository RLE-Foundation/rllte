import torch
from torch import nn
from torch.nn import functional as F

from hsuanwu.common.typing import Distribution, Space, Tensor, Tuple
from hsuanwu.xploit.learner import utils


class StochasticActor(nn.Module):
    """Stochastic actor network for SACLearner. Here the 'self.dist' refers to an sampling distribution instance.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network instance.
    """

    def __init__(
        self,
        action_space: Space,
        feature_dim: int = 64,
        hidden_dim: int = 1024,
        log_std_range: Tuple = (-10, 2),
    ) -> None:
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_space.shape[0]),
        )
        # placeholder for distribution
        self.dist = None
        self.log_std_min, self.log_std_max = log_std_range

        self.apply(utils.network_init)

    def get_action(self, obs: Tensor, step: float = None) -> Distribution:
        """Get actions.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Hsuanwu distribution.
        """
        mu, log_std = self.policy(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        std = log_std.exp()

        return self.dist(mu, std)


class DeterministicActor(nn.Module):
    """Deterministic actor network for DrQv2Learner. Here the 'self.dist' refers to an action noise instance.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network instance.
    """

    def __init__(
        self, action_space: Space, feature_dim: int = 64, hidden_dim: int = 1024
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space.shape[0]),
        )
        # placeholder for distribution
        self.dist = None

        self.apply(utils.network_init)

    def get_action(self, obs: Tensor, step: float = None) -> Distribution:
        """Get actions.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Hsuanwu distribution.
        """
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)

        # for Scheduled Exploration Noise
        self.dist.reset(mu, step)

        return self.dist


class DoubleCritic(nn.Module):
    """Double critic network for DrQv2Learner and SACLearner.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Critic network instance.
    """

    def __init__(
        self, action_space: Space, feature_dim: int = 64, hidden_dim: int = 1024
    ) -> None:
        super().__init__()

        action_shape = action_space.shape
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.network_init)

    def forward(self, obs: Tensor, action: Tensor) -> Tuple[Tensor]:
        """Value estimation.

        Args:
            obs (Tensor): Observations.
            action (Tensor): Actions.

        Returns:
            Estimated values.
        """
        h_action = torch.cat([obs, action], dim=-1)

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DiscreteActorCritic(nn.Module):
    """Actor-Critic network for discrete control tasks. For PPOLearner, DrACLearner.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor-Critic instance.
    """

    def __init__(self, action_space: Space, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_space.shape[0])
        self.critic = nn.Linear(hidden_dim, 1)
        # placeholder for distribution
        self.dist = None

        self.apply(utils.network_init)

    def get_value(self, obs: Tensor) -> Tensor:
        """Get estimated values for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        return self.critic(self.trunk(obs))

    def get_action(self, obs: Tensor) -> Tensor:
        """Get deterministic actions for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        mu = self.actor(self.trunk(obs))
        return self.dist(mu).mode

    def get_action_and_value(
        self, obs: Tensor, actions: Tensor = None
    ) -> Tuple[Tensor]:
        """Get actions and estimated values for observations.

        Args:
            obs (Tensor): Sampled observations.
            actions (Tensor): Sampled actions.

        Returns:
            Actions, Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        h = self.trunk(obs)
        mu = self.actor(h)
        dist = self.dist(mu)
        if actions is None:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return actions, self.critic(h), log_probs, entropy


class DiscreteActorAuxiliaryCritic(nn.Module):
    """Actor-Critic network for discrete control tasks. For PPGLearner.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor-Critic instance.
    """

    def __init__(self, action_space: Space, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_space.shape[0])
        self.critic = nn.Linear(hidden_dim, 1)
        self.aux_critic = nn.Linear(hidden_dim, 1)
        # placeholder for distribution
        self.dist = None

        self.apply(utils.network_init)

    def get_value(self, obs: Tensor) -> Tensor:
        """Get estimated values for observations.

        Args:
            obs: Observations.

        Returns:
            Estimated values.
        """
        return self.critic(self.trunk(obs))

    def get_action(self, obs: Tensor) -> Tensor:
        """Get deterministic actions for observations.

        Args:
            obs: Observations.

        Returns:
            Estimated values.
        """
        logits = self.actor(self.trunk(obs))
        return self.dist(logits).mode

    def get_action_and_value(
        self, obs: Tensor, actions: Tensor = None
    ) -> Tuple[Tensor]:
        """Get actions and estimated values for observations.

        Args:
            obs: Sampled observations.
            actions: Sampled actions.

        Returns:
            Actions, Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        h = self.trunk(obs)
        logits = self.actor(h)
        dist = self.dist(logits)
        if actions is None:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return actions, self.critic(h), log_probs, entropy

    def get_probs_and_aux_value(self, obs: Tensor) -> Tuple[Tensor]:
        """Get probs and auxiliary estimated values for auxiliary phase update.

        Args:
            obs: Sampled observations.

        Returns:
            Distribution, estimated values, auxiliary estimated values.
        """
        h = self.trunk(obs)
        logits = self.actor(h)
        dist = self.dist(logits)

        return dist, self.critic(h.detach()), self.aux_critic(h)

    def get_logits(self, obs: Tensor) -> Distribution:
        """Get the log-odds of sampling.

        Args:
            obs: Sampled observations.

        Returns:
            Distribution
        """
        return self.dist(self.actor(self.trunk(obs)))
