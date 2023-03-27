import torch

from hsuanwu.common.typing import *
from hsuanwu.xplore.augmentation.base import BaseAugmentation

class RandomTranslate(BaseAugmentation):
    
    """Random translate operation for processing image-based observations.
    Args:
        size: the scale size in translated images

    Returns:
        Random translate images.
    """
    
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, imgs):
        n, c, h, w = imgs.shape
        device = imgs.device
        assert self.size >= h and self.size >= w
        outs = torch.zeros((n, c, self.size, self.size), dtype=imgs.dtype, device=device)
        h1s = torch.randint(0, self.size - h + 1, (n,), device=device)
        w1s = torch.randint(0, self.size - w + 1, (n,), device=device)
        for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
            out[:, h1:h1 + h, w1:w1 + w] = img
        return outs