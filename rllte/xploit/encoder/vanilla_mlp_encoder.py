from typing import Union

import gymnasium as gym
import torch as th
from omegaconf import DictConfig
from torch import nn

from hsuanwu.common.base_encoder import BaseEncoder


class VanillaMlpEncoder(BaseEncoder):
    """Multi layer perceptron (MLP) for processing state-based inputs.

    Args:
        observation_space (Space): The observation space of environment.
        feature_dim (int): Number of features extracted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Mlp-based encoder instance.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        feature_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(observation_space, feature_dim)

        input_dim = observation_space.shape[0]
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, feature_dim), nn.Tanh())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.trunk(obs)
