import torch as th
from torch import nn
from abc import abstractmethod

class BaseAugmentation(nn.Module):
    """Base class of augmentation."""


    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(*args) -> th.Tensor:
        """Augmentation function.
        """