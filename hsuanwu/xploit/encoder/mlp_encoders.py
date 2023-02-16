from torch import nn

from hsuanwu.common.typing import *
from hsuanwu.xploit.encoder.base import BaseEncoder
from hsuanwu.xploit.utils import network_init



class MlpEncoder(BaseEncoder):
    """
    Multi layer perceptron (MLP) for processing state-based inputs.

    :param observation_space: Observation space of the environment.
    :param features_dim: Number of features extracted.
    :param hidden_dim: Number of units per hidden layer.
    """
    def __init__(self, observation_space: Space, features_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, features_dim)
        )

        self.apply(network_init)

    def forward(self, obs: Tensor) -> Tensor:
        return self.trunk(obs)