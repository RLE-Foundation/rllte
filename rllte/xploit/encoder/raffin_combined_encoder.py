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
from rllte.common.preprocessing import get_flattened_obs_dim, preprocess_obs
from rllte.xploit.encoder.pathak_cnn_encoder import PathakCnnEncoder


class RaffinCombinedEncoder(BaseEncoder):
    """Combined features extractor for Dict observation spaces.
        Based on: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L231

    Args:
        observation_space (gym.Space): Observation space.
        feature_dim (int): Number of features extracted.
        cnn_output_dim (int): Number of features extracted by the CNN.

    Returns:
        Identity encoder instance.
    """

    def __init__(self, observation_space: gym.Space, feature_dim: int = 256, cnn_output_dim: int = 256) -> None:
        super().__init__(observation_space, feature_dim)

        sub_encoders = dict()
        n_flatten = 0

        for key, subspace in observation_space.spaces.items():
            if len(subspace.shape) > 1:
                sub_encoders[key] = PathakCnnEncoder(subspace, feature_dim=cnn_output_dim)
                n_flatten += cnn_output_dim
            else:
                sub_encoders[key] = nn.Identity()
                n_flatten += get_flattened_obs_dim(subspace)

        self.trunk = nn.ModuleDict(sub_encoders)
        self.linear = nn.Linear(n_flatten, feature_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Forward method implementation.

        Args:
            obs (th.Tensor): Observation tensor.

        Returns:
            Encoded observation tensor.
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space)
        encoded_obs = []

        for key, sub_encoder in self.trunk.items():
            encoded_obs.append(sub_encoder(preprocessed_obs[key]))

        return self.linear(th.cat(encoded_obs, dim=1))
