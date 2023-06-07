from typing import Tuple
from pathlib import Path
import os
import torch as th
from torch import nn
from torch.distributions import Distribution
from rllte.common.utils import ExportModel
from rllte.common.policies.off_policy_deterministic_actor_double_critic import DoubleCritic

class OffPolicyStochasticActorDoubleCritic(nn.Module):
    """Stochastic actor network and double critic network for SAC. 
        Here the 'self.dist' refers to an sampling distribution instance.

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

        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_dim),
        )

        self.critic = DoubleCritic(
            action_dim=action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )

        self.critic_target = DoubleCritic(
            action_dim=action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        # synchronize critic and target critic
        self.critic_target.load_state_dict(self.critic.state_dict())

        # placeholder for distribution
        self.encoder = None
        self.dist = None
        self.log_std_min, self.log_std_max = log_std_range
    
    def forward(self, obs: th.Tensor, training: bool = True, step: int = 0) -> Tuple[th.Tensor]:
        """Sample actions based on observations.
        
        Args:
            obs (Tensor): Observations.
            training (bool): Training mode, True or False.
            step (int): Global training step.

        Returns:
            Sampled actions.
        """
        encoded_obs = self.encoder(obs)
        dist = self.get_dist(obs=encoded_obs, step=step)

        if not training:
            action = dist.mean
        else:
            action = dist.sample()

        return action

    def get_dist(self, obs: th.Tensor, step: int) -> Distribution:
        """Get sample distribution.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            RLLTE distribution.
        """
        mu, log_std = self.actor(obs).chunk(2, dim=-1)

        log_std = th.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = log_std.exp()

        return self.dist(mu, std)

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

class NpuOffPolicyStochasticActorDoubleCritic(OffPolicyStochasticActorDoubleCritic):
    """Stochastic actor network and double critic network for SAC and `NPU` device. 
        Here the 'self.dist' refers to an sampling distribution instance.

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
        super().__init__(action_dim=action_dim, feature_dim=feature_dim, hidden_dim=hidden_dim, log_std_range=log_std_range)
    
    def forward(self, obs: th.Tensor, training: bool = True, step: int = 0) -> Tuple[th.Tensor]:
        """Sample actions based on observations.
        
        Args:
            obs (Tensor): Observations.
            training (bool): Training mode, True or False.
            step (int): Global training step.

        Returns:
            Sampled actions.
        """
        encoded_obs = self.encoder(obs)
        dist = self.get_dist(obs=encoded_obs, step=step)

        if not training:
            action = dist.mean
        else:
            action = dist.sample()

        return action

    def get_dist(self, obs: th.Tensor, step: int) -> Distribution:
        """Get sample distribution, for `NPU` device.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            RLLTE distribution.
        """
        mu, log_std = self.actor(obs).chunk(2, dim=-1)

        log_std = th.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = log_std.exp()

        return self.dist(mu.cpu(), std.cpu())
