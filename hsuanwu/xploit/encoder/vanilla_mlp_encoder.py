from typing import Union, Dict, List
import gymnasium as gym
from omegaconf import DictConfig
import torch as th
from torch import nn

from hsuanwu.xploit.encoder.base import BaseEncoder, network_init


class VanillaMlpEncoder(BaseEncoder):
    """Multi layer perceptron (MLP) for processing state-based inputs.

    Args:
        obs_space (Space or DictConfig): The observation space of environment. When invoked by Hydra, 
            'obs_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        feature_dim (int): Number of features extracted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Mlp-based encoder instance.
    """

    def __init__(
        self, obs_space: Union[gym.Space, DictConfig], feature_dim: int = 64, hidden_dim: int = 256
    ) -> None:
        super().__init__(obs_space, feature_dim)

        input_dim = obs_space.shape[0]
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.apply(network_init)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.trunk(obs)
