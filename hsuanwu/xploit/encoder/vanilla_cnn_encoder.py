from torch import nn

from hsuanwu.common.typing import *
from hsuanwu.xploit.encoder.base import BaseEncoder
from hsuanwu.xploit.utils import network_init


class VanillaCnnEncoder(BaseEncoder):
    """Convolutional neural network (CNN)-based encoder for processing image-based observations.

    Args:
        observation_space: Observation space of the environment.
        feature_dim: Number of features extracted.

    Returns:
        CNN-based encoder instance.
    """

    def __init__(self, observation_space: Space, feature_dim: int = 50) -> None:
        super().__init__(observation_space, feature_dim)

        obs_shape = observation_space.shape
        assert len(obs_shape) == 3
        self.trunk = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.ones(size=tuple(obs_shape)).float()
            n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

        self.linear = nn.Linear(n_flatten, feature_dim)

        self.apply(network_init)

    def forward(self, obs: Tensor) -> Tensor:
        obs = obs / 255.0 - 0.5
        h = self.trunk(obs)

        return self.linear(h.view(h.size()[0], -1))
