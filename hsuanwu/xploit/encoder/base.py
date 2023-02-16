from torch import nn

from hsuanwu.common.typing import *


class BaseEncoder(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space: Observation space of the environment.
    """

    def __init__(self, observation_space: Space, features_dim: int = 0) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim