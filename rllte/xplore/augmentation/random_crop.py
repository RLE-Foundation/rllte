import torch as th
from torch.nn import functional as F

from hsuanwu.common.base_augmentation import BaseAugmentation


class RandomCrop(BaseAugmentation):
    """Random crop operation for processing image-based observations.

    Args:
        pad (int): Padding size.
        out (int): Desired output size.

    Returns:
        Augmented images.
    """

    def __init__(self, pad: int = 4, out: int = 84) -> None:
        super().__init__()
        self._out = out
        self._pad = pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self._pad] * 4)
        x = F.pad(x, padding, "replicate")

        crop_max = x.size()[2] - self._out + 1
        new_w = th.randint(0, crop_max, (n,))
        new_h = th.randint(0, crop_max, (n,))
        cropped = th.empty(size=(n, c, self._out, self._out), device=x.device)

        for idx, (img, t_h, t_w) in enumerate(zip(x, new_h, new_w)):
            cropped[idx] = img[:, t_h : t_h + self._out, t_w : t_w + self._out]

        return cropped
