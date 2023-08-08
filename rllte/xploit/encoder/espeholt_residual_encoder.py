# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from typing import List

import gymnasium as gym
import torch as th
from torch import nn
from torch.nn import functional as F

from rllte.common.prototype import BaseEncoder


class Conv2d_tf(nn.Conv2d):
    """Conv2d with the padding behavior from TF."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class ResidualBlock(nn.Module):
    """Residual block based on
        https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py

    Args:
        n_channels (int): Channels of inputs.

    Returns:
        Single residual block.
    """

    def __init__(self, n_channels, stride=1):
        super().__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ResidualLayer(nn.Module):
    """Single residual layer for building ResNet encoder.

    Args:
        in_channels (int): Channels of inputs.
        out_channels (int): Channels of outputs.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(ResidualBlock(out_channels))
        layers.append(ResidualBlock(out_channels))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class EspeholtResidualEncoder(BaseEncoder):
    """ResNet-like encoder for processing image-based observations.
        Proposed by Espeholt L, Soyer H, Munos R, et al. Impala: Scalable distributed deep-rl with importance
        weighted actor-learner architectures[C]//International conference on machine learning. PMLR, 2018: 1407-1416.
        Target task: Atari games and Procgen games.

    Args:
        observation_space (gym.Space): Observation space.
        feature_dim (int): Number of features extracted.
        net_arch (List): Architecture of the network.
            It represents the out channels of each residual layer.
            The length of this list is the number of residual layers.

    Returns:
        ResNet-like encoder instance.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        feature_dim: int = 0,
        net_arch: List[int] = [16, 32, 32],  # noqa B008
    ) -> None:
        super().__init__(observation_space, feature_dim)
        assert len(net_arch) >= 1, "At least one Residual layer!"
        modules = list()
        shape = observation_space.shape
        if len(shape) == 4:
            # vectorized envs
            shape = shape[1:]

        in_channels = shape[0]
        for out_channels in net_arch:
            layer = ResidualLayer(in_channels, out_channels)
            modules.append(layer)
            in_channels = out_channels

        modules.append(nn.Flatten())
        self.trunk = nn.Sequential(*modules)
        with th.no_grad():
            sample = th.ones(size=tuple(shape)).float()
            n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

        self.trunk.append(nn.Linear(n_flatten, feature_dim))
        self.trunk.append(nn.ReLU())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Forward method implementation.

        Args:
            obs (th.Tensor): Observation tensor.

        Returns:
            Encoded observation tensor.
        """
        h = self.trunk(obs / 255.0)
        return h
