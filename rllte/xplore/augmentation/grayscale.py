import torch as th

from rllte.common.base_augmentation import BaseAugmentation


class GrayScale(BaseAugmentation):
    """Grayscale operation for image augmentation.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, h, w = x.size()
        frames = c // 3
        x = x.view([b, frames, 3, h, w])
        x = 0.2989 * x[:, :, 0, ...] + 0.5870 * x[:, :, 1, ...] + 0.1140 * x[:, :, 2, ...]
        return x
