# =============================================================================
# MIT License

# Copyright (c) 2024 Reinforcement Learning Evolution Foundation

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


from typing import Tuple
from torch import nn

import torch as th

class CnnObservationEncoder(nn.Module):
    """CNN-based encoder for encoding observations.

    Args:
        obs_shape (Tuple): The data shape of observations.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
            )
        with th.no_grad():
            sample = th.ones(size=tuple(obs_shape))
            n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

        self.linear = nn.Linear(n_flatten, latent_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return self.linear(self.trunk(obs / 255.0))

class MlpObservationEncoder(nn.Module):
    """MLP-based encoder for encoding observations.

    Args:
        obs_shape (Tuple): The data shape of observations.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(obs_shape[0], 256), nn.ReLU())
        self.linear = nn.Linear(256, latent_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return self.linear(self.trunk(obs))