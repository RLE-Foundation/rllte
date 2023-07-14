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


from typing import Callable, Tuple, Dict

import json
import numpy as np
import torch as th
import gymnasium as gym

from torch import nn


class ExportModel(nn.Module):
    """Module for model export.

    Args:
        encoder (nn.Module): Encoder network.
        actor (nn.Module): Actor network.

    Returns:
        Export model format.
    """

    def __init__(self, encoder: nn.Module, actor: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.actor = actor

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        return self.actor(self.encoder(obs))


class eval_mode:
    """Set the evaluation mode.

    Args:
        models (nn.Module): Models.

    Returns:
        None.
    """

    def __init__(self, *models) -> None:
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.mode(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.mode(state)
        return False


def get_network_init(method: str = "orthogonal") -> Callable:  # noqa: c901
    """Returns a network initialization function.

    Args:
        method (str): Initialization method name.

    Returns:
        Initialization function.
    """

    def _identity(m):
        """Identity initialization."""
        pass

    def _orthogonal(m):
        """Orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(m.weight.data, gain)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    def _xavier_uniform(m):
        """Xavier uniform initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    def _xavier_normal(m):
        """Xavier normal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    if method == "orthogonal":
        return _orthogonal
    elif method == "xavier_normal":
        return _xavier_normal
    elif method == "xavier_uniform":
        return _xavier_uniform
    else:
        return _identity


def to_numpy(xs: Tuple[th.Tensor, ...]) -> Tuple[np.ndarray, ...]:
    """Converts torch tensors to numpy arrays.

    Args:
        xs (Tuple[th.Tensor, ...]): Torch tensors.

    Returns:
        Numpy arrays.
    """
    for x in xs:
        print(x.size())
    return tuple(x[0].cpu().numpy() for x in xs)


def pretty_json(hp: Dict) -> str:
    """Returns a pretty json string.

    Args:
        hp (Dict): Hyperparameters.

    Returns:
        Pretty json string.
    """
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


def process_env_info(observation_space: gym.Space, action_space: gym.Space) -> Tuple[Tuple, ...]:
    """Process environment information.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.

    Returns:
        Observation and action space information.
    """
    obs_shape = observation_space.shape
    if action_space.__class__.__name__ == "Discrete":
        action_shape = action_space.shape
        action_dim = int(action_space.n)
        action_type = "Discrete"
        action_range = [0, int(action_space.n) - 1]
    elif action_space.__class__.__name__ == "Box":
        action_shape = action_space.shape
        action_dim = action_space.shape[0]
        action_type = "Box"
        action_range = [
            float(action_space.low[0]),
            float(action_space.high[0]),
        ]
    elif action_space.__class__.__name__ == "MultiBinary":
        action_shape = action_space.shape
        action_dim = action_space.shape[0]
        action_type = "MultiBinary"
        action_range = [0, 1]
    else:
        raise NotImplementedError("Unsupported action type!")
    
    return obs_shape, action_shape, action_dim, action_type, action_range