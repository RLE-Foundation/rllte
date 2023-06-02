from typing import Tuple

import torch as th
from torch import nn
from torch.distributions import Distribution


class OffPolicyStochasticActor(nn.Module):
    """Stochastic actor network for SAC. Here the 'self.dist' refers to an sampling distribution instance.

    Args:
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network instance.
    """

    def __init__(
        self,
        action_dim: int,
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
            nn.Linear(hidden_dim, 2 * action_dim),
        )
        # placeholder for distribution
        self.dist = None
        self.log_std_min, self.log_std_max = log_std_range

    def get_dist(self, obs: th.Tensor, step: int) -> Distribution:
        """Get sample distribution.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Hsuanwu distribution.
        """
        mu, log_std = self.policy(obs).chunk(2, dim=-1)

        log_std = th.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = log_std.exp()

        return self.dist(mu, std)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Get deterministic actions.

        Args:
            obs (Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        mu, _ = self.policy(obs).chunk(2, dim=-1)
        return mu
