from torch import nn

from hsuanwu.common.typing import Space, Tensor
from hsuanwu.xploit.encoder.base import BaseEncoder
from hsuanwu.xploit.utils import network_init


class IdentityEncoder(BaseEncoder):
    """Identity encoder for state-based observations.

    Args:
        observation_space: Observation space of the environment.
        feature_dim: Number of features extracted.

    Returns:
        Identity encoder instance.
    """

    def __init__(self, observation_space: Space, feature_dim: int = 64) -> None:
        super().__init__(observation_space, feature_dim)

        obs_shape = observation_space.shape
        assert len(obs_shape) == 1
        self.trunk = nn.Sequential(nn.Linear(1, 1))

    def forward(self, obs: Tensor) -> Tensor:
        return obs
