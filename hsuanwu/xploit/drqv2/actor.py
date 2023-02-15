from torch import nn

from hsuanwu.common.typing import *
from hsuanwu.xploit.utils import network_init
from hsuanwu.xploit.encoders import CnnEncoder, MlpEncoder

class Actor(nn.Module):
    """
    Actor network

    :param action_space: Action space of the environment.
    :param features_dim: Number of features accepted.
    :param hidden_dim: Number of units per hidden layer.
    """
    def __init__(self, action_space: Space, features_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(features_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_space.shape[0]))
    
        self.apply(network_init)
    
