from typing import Union, Dict
import gymnasium as gym
from omegaconf import DictConfig

from torch import nn


def network_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class BaseEncoder(nn.Module):
    """Base class that represents a features extractor.

    Args:
        obs_space (Space or DictConfig): The observation space of environment. When invoked by Hydra, 
            'obs_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        feature_dim (int): Number of features extracted.

    Returns:
        The base encoder class
    """

    def __init__(self, obs_space: Union[gym.Space, DictConfig], feature_dim: int = 0) -> None:
        super().__init__()
        assert feature_dim > 0
        self._obs_space = obs_space
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim
