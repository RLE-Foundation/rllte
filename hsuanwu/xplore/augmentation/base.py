import torch as th
from torch import nn


class BaseAugmentation(nn.Module):
    """Base class of augmentation."""

    def __init__(self) -> None:
        super().__init__()

    def forward(*args) -> th.Tensor:
        """Augmentation function."""
