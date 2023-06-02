from typing import Union

import gymnasium as gym
import torch as th
from omegaconf import DictConfig
from torch import nn

from rllte.common.base_encoder import BaseEncoder


class IdentityEncoder(BaseEncoder):
    """Identity encoder for state-based observations.

    Args:
        observation_space (Space): The observation space of environment.
        feature_dim (int): Number of features extracted.

    Returns:
        Identity encoder instance.
    """

    def __init__(self, observation_space: gym.Space, feature_dim: int = 64) -> None:
        super().__init__(observation_space, feature_dim)

        obs_shape = observation_space.shape
        assert len(obs_shape) == 1
        self.trunk = nn.Sequential(nn.Identity(obs_shape[0]))
        self.unused = nn.Linear(1, 1)  # for avoiding the ValueError: optimizer got an empty parameter list

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.trunk(obs)
