import torch
from hsuanwu.common.typing import *
from hsuanwu.xplore.augmentation.base import BaseAugmentation
from torchvision.transforms import ColorJitter

class RandomColorJitter(BaseAugmentation):
    """Random ColorJitter operation for image augmentation.

    Args:
        brightness: How much to jitter brightness. Should be non negative numbers.
        contrast: How much to jitter contrast. Should be non negative numbers.
        saturation: How much to jitter saturation. Should be non negative numbers.
        hue: How much to jitter hue. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. 
    
    Returns:
        Augmented images.
    """
    def __init__(self, 
                 brightness: float = 0.4,
                 contrast: float = 0.4, 
                 saturation: float = 0.4, 
                 hue: float = 0.5) -> None:
        super(RandomColorJitter, self).__init__()
        self.color_jitter = ColorJitter(brightness=brightness, 
                                        contrast=contrast,
                                        saturation=saturation, 
                                        hue=hue)

    def forward(self, x: Tensor):
        b, c, h, w = x.size()
        x = x.view(-1, 3, h, w)
        x = self.color_jitter(x)
        x = x.view(b, c, h, w)
        return x
