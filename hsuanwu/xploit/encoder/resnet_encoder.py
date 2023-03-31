import numpy as np
from torch import nn
from torch.nn import functional as F

from hsuanwu.common.typing import List, Space, Tensor, Tuple
from hsuanwu.xploit.encoder.base import BaseEncoder
from hsuanwu.xploit.utils import network_init


class ResidualBlock(nn.Module):
    """Residual block taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py

    Args:
        channels: Channels of inputs.
    """

    def __init__(self, channels: List) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)

        return x + inputs


class ResidualLayer(nn.Module):
    """Single residual layer for building ResNet encoder.

    Args:
        input_shape: Data shape of the inputs.
        out_channels: Channels of outputs.
    """

    def __init__(self, input_shape: Tuple, out_channels: int):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ResNetEncoder(BaseEncoder):
    """
    ResNet-like encoder for processing image-based observations.

    Args:
        observation_space (Space): Observation space of the environment.
        feature_dim (int): Number of features extracted.
        net_arch (List): Architecture of the network.
            It represents the out channels of each residual layer.
            The length of this list is the number of residual layers.

    Returns:
        ResNet-like encoder instance.
    """

    def __init__(
        self,
        observation_space: Space,
        feature_dim: int = 0,
        net_arch: List[int] = [16, 32, 32],
    ) -> None:
        super().__init__(observation_space, feature_dim)
        assert len(net_arch) >= 1, "At least one Residual layer!"
        modules = list()
        shape = observation_space.shape
        if len(shape) == 4:
            # vectorized envs
            shape = shape[1:]

        for out_channels in net_arch:
            layer = ResidualLayer(shape, out_channels)
            shape = layer.get_output_shape()
            modules.append(layer)
        modules.append(nn.Flatten())

        self.trunk = nn.Sequential(*modules)
        self.linear = nn.Linear(
            in_features=shape[0] * shape[1] * shape[2], out_features=feature_dim
        )

        self.apply(network_init)

    def forward(self, obs: Tensor) -> Tensor:
        obs = obs / 255.0
        h = self.trunk(obs)
        return self.linear(h)
