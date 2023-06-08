from typing import Callable

import torch as th
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
            obs (Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        return self.actor(self.encoder(obs))


class eval_mode:
    """Set the evaluation mode."""

    def __init__(self, *models):
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
