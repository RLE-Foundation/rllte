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


import gymnasium as gym
import torch as th
from torch import nn

from rllte.common.prototype import BaseEncoder


class IdentityEncoder(BaseEncoder):
    """Identity encoder for state-based observations.

    Args:
        observation_space (gym.Space): Observation space.
        feature_dim (int): Number of features extracted.

    Returns:
        Identity encoder instance.
    """

    def __init__(self, observation_space: gym.Space, feature_dim: int = 64) -> None:
        super().__init__(observation_space, feature_dim)

        obs_shape = observation_space.shape
        assert len(obs_shape) == 1
        self.trunk = nn.Sequential(nn.Identity(obs_shape[0]))
        self.unused = nn.Linear(1, 1)  # for avoiding the ValueError: optimizer got an empty parameter list

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Forward method implementation.

        Args:
            obs (th.Tensor): Observation tensor.

        Returns:
            Encoded observation tensor.
        """
        return self.trunk(obs)
