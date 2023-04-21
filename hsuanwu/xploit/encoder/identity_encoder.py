from typing import Union, Dict
import gymnasium as gym
from omegaconf import DictConfig
import torch as th
from torch import nn

from hsuanwu.xploit.encoder.base import BaseEncoder, network_init


class IdentityEncoder(BaseEncoder):
    """Identity encoder for state-based observations.

    Args:
        obs_space (Space or DictConfig): The observation space of environment. When invoked by Hydra, 
            'obs_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        feature_dim (int): Number of features extracted.

    Returns:
        Identity encoder instance.
    """

    def __init__(self, obs_space: Union[gym.Space, DictConfig], feature_dim: int = 64) -> None:
        super().__init__(obs_space, feature_dim)

        obs_shape = obs_space.shape
        assert len(obs_shape) == 1
        self.trunk = nn.Sequential(nn.Linear(1, 1))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return obs
