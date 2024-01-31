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
from torch.nn import functional as F

import torch as th

class ObservationEncoder(nn.Module):
    """Encoder for encoding observations.

    Args:
        obs_shape (Tuple): The data shape of observations.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super().__init__()

        # visual
        if len(obs_shape) == 3:
            self.trunk = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with th.no_grad():
                sample = th.ones(size=tuple(obs_shape)).float()
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.trunk.append(nn.Linear(n_flatten, latent_dim))
        else:
            self.trunk = nn.Sequential(nn.Linear(obs_shape[0], 256), 
                                       nn.ReLU())
            self.trunk.append(nn.Linear(256, latent_dim))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return self.trunk(obs)
    
class InverseDynamicsEncoder(nn.Module):
    """Encoder with inverse dynamics prediction.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): The dimension of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.encoder = ObservationEncoder(obs_shape, latent_dim)
        self.policy = InverseDynamicsModel(latent_dim, action_dim)

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted actions.

        Args:
            obs (th.Tensor): Current observations.
            next_obs (th.Tensor): Next observations.

        Returns:
            Predicted actions.
        """
        h = self.encoder(obs)
        next_h = self.encoder(next_obs)

        actions = self.policy(h, next_h)
        return actions

    def encode(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return self.encoder(obs)
    

class InverseDynamicsModel(nn.Module):
    """Inverse model for reconstructing transition process.

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """

    def __init__(self, latent_dim, action_dim) -> None:
        super().__init__()

        self.trunk = ObservationEncoder(obs_shape=(latent_dim * 2,), latent_dim=action_dim)

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted actions.

        Args:
            obs (th.Tensor): Current observations.
            next_obs (th.Tensor): Next observations.

        Returns:
            Predicted actions.
        """
        return self.trunk(th.cat([obs, next_obs], dim=1))

class ForwardDynamicsModel(nn.Module):
    """Forward model for reconstructing transition process.

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """

    def __init__(self, latent_dim, action_dim) -> None:
        super().__init__()

        self.trunk = ObservationEncoder(obs_shape=(latent_dim + action_dim,), latent_dim=latent_dim)

    def forward(self, obs: th.Tensor, pred_actions: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted next-obs.

        Args:
            obs (th.Tensor): Current observations.
            pred_actions (th.Tensor): Predicted observations.

        Returns:
            Predicted next-obs.
        """
        print(obs.shape, pred_actions.shape)
        return self.trunk(th.cat([obs, pred_actions], dim=1))