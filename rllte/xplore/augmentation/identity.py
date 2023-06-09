import torch as th
from rllte.common.base_augmentation import BaseAugmentation

class Identity(BaseAugmentation):
    """Identity augmentation.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x
