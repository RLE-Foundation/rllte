from typing import Tuple
from pathlib import Path
import os
import torch as th
from torch import nn
from torch.distributions import Distribution
from rllte.common.utils import ExportModel

class DoubleCritic(nn.Module):
    """Double critic network for DrQv2 and SAC.

    Args:
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Critic network instance.
    """

    def __init__(self, action_dim: int, feature_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, ...]:
        """Value estimation.

        Args:
            obs (Tensor): Observations.
            action (Tensor): Actions.

        Returns:
            Estimated values.
        """
        h_action = th.cat([obs, action], dim=-1)

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class OffPolicyDeterministicActorDoubleCritic(nn.Module):
    """Deterministic actor network and double critic network for DrQv2. Here the 'self.dist' refers to an action noise instance.

    Args:
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network instance.
    """
    def __init__(self, action_dim: int, feature_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
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
        mu = self.actor(obs)

        # for Scheduled Exploration Noise
        self.dist.reset(mu, step)

        return self.dist

    def get_dist_npu(self, obs: th.Tensor, step: int) -> Distribution:
        """Get sample distribution, for `NPU` device.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            RLLTE distribution.
        """
        mu = self.actor(obs)

        # for Scheduled Exploration Noise
        self.dist.reset(mu.cpu(), step)

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

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        params = th.load(os.path.join(path, "pretrained.pth"), map_location=self.device)
        self.load_state_dict(params)

class NpuOffPolicyDeterministicActorDoubleCritic(OffPolicyDeterministicActorDoubleCritic):
    """Deterministic actor network and double critic network for DrQv2 and `NPU` device. 
        Here the 'self.dist' refers to an action noise instance.

    Args:
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network instance.
    """
    def __init__(self, action_dim: int, feature_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__(action_dim=action_dim, feature_dim=feature_dim, hidden_dim=hidden_dim)
        
    
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
        mu = self.actor(obs)

        # for Scheduled Exploration Noise
        self.dist.reset(mu.cpu(), step)

        return self.dist