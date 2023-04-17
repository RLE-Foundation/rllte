import torch
import torch.nn.functional as F

from hsuanwu.common.typing import *
from hsuanwu.xplore.augmentation.base import BaseAugmentation


class RandomCrop(BaseAugmentation):
    """Random crop operation for processing image-based observations.

    Args:
        pad: Padding size.
        out: Desired output size.

    Returns:
        Augmented images.
    """

    def __init__(self, pad: int = 4, out: int = 84) -> None:
        super().__init__()
        self._out = out
        self._pad = pad

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self._pad] * 4)
        x = F.pad(x, padding, "replicate")

        crop_max = x.size()[2] - self._out + 1
        new_w = torch.randint(0, crop_max, (n,))
        new_h = torch.randint(0, crop_max, (n,))
        cropped = torch.empty(size=(n, c, self._out, self._out))

        for idx, (img, new_h, new_w) in enumerate(zip(x, new_h, new_w)):
            cropped[idx] = img[:, new_h : new_h + self._out, new_w : new_w + self._out]

        return cropped
