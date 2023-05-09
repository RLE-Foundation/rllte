from typing import Union

import gymnasium as gym
import torch as th
from omegaconf import DictConfig
from torch import nn

from hsuanwu.xploit.encoder.base import BaseEncoder


class MnihCnnEncoder(BaseEncoder):
    """Convolutional neural network (CNN)-based encoder for processing image-based observations.
    Proposed by Mnih V, Kavukcuoglu K, Silver D, et al. Playing atari with
    deep reinforcement learning[J]. arXiv preprint arXiv:1312.5602, 2013.
    Target task: Atari games.

    Args:
        observation_space (Space or DictConfig): The observation space of environment. When invoked by Hydra,
            'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        feature_dim (int): Number of features extracted.

    Returns:
        CNN-based encoder instance.
    """

    def __init__(self, observation_space: Union[gym.Space, DictConfig], feature_dim: int = 0) -> None:
        super().__init__(observation_space, feature_dim)

        obs_shape = observation_space.shape
        assert len(obs_shape) == 3

        self.trunk = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.ones(size=tuple(obs_shape)).float()
            n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

        self.trunk.extend([nn.Linear(n_flatten, feature_dim), nn.ReLU()])

    def forward(self, obs: th.Tensor) -> th.Tensor:
        h = self.trunk(obs / 255.0)

        return h.view(h.size()[0], -1)
