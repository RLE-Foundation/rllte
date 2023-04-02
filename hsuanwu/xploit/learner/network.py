from torch import nn
from torch.nn import functional as F

import torch
from hsuanwu.common.typing import Tensor, Distribution, Space, Tuple
from hsuanwu.xploit import utils

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

