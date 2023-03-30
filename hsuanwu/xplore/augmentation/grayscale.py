import torch

from hsuanwu.common.typing import *
from hsuanwu.xplore.augmentation.base import BaseAugmentation


class GrayScale(BaseAugmentation):
    """Grayscale operation for image augmentation.

    Args:
        None.

    Returns:
        Augmented images.
    """

    def __init__(self) -> None:
        super(GrayScale, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()
        frames = c // 3
        x = x.view([b, frames, 3, h, w])
        x = (
            0.2989 * x[:, :, 0, ...]
            + 0.5870 * x[:, :, 1, ...]
            + 0.1140 * x[:, :, 2, ...]
        )
        return x
