from torch import nn

from hsuanwu.common.typing import *
from hsuanwu.xploit.utils import network_init

class Critic(nn.Module):
    """
    Critic network

    :param action_space: Action space of the environment.
    :param features_dim: Number of features accepted.
    :param hidden_dim: Number of units per hidden layer.
    """
    def __init__(self, action_space: Space, features_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(features_dim), nn.Tanh())

        action_shape = action_space.shape
        self.Q1 = nn.Sequential(
            nn.Linear(features_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(features_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(network_init)
    
    def forward(self, obs: Tensor, action: Tensor):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
