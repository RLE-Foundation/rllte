from typing import Tuple

import torch as th
from torch import nn
from torch.distributions import Distribution

class OffPolicyDeterministicActor(nn.Module):
    """Deterministic actor network for DrQv2. Here the 'self.dist' refers to an action noise instance.

    Args:
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network instance.
    """

    def __init__(self, 
                 action_dim: int,
                 feature_dim: int = 64, 
                 hidden_dim: int = 1024) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # placeholder for distribution
        self.dist = None

    def get_dist(self, obs: th.Tensor, step: int) -> Distribution:
        """Get sample distribution.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Hsuanwu distribution.
        """
        mu = self.policy(obs)

        # for Scheduled Exploration Noise
        self.dist.reset(mu, step)

        return self.dist

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Get actions.

        Args:
            obs (Tensor): Observations.

        Returns:
            Actions.
        """
        return self.policy(obs)