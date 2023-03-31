from torch import nn

from hsuanwu.common.typing import Space, Tensor


class BaseEncoder(nn.Module):
    """Base class that represents a features extractor.

    Args:
        observation_space (Space): Observation space of the environment.
        feature_dim (int): Number of features extracted.

    Returns:
        The base encoder class
    """

    def __init__(self, observation_space: Space, feature_dim: int = 0) -> None:
        super().__init__()
        assert feature_dim > 0
        self._observation_space = observation_space
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim
