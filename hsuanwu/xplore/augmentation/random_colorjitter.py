import torch
from hsuanwu.common.typing import *
from hsuanwu.xplore.augmentation.base import BaseAugmentation
# From torch vision import the colorjitter
from torchvision.transforms import ColorJitter

class RandomColorJitter(BaseAugmentation):
    
    """Random ColorJitter operation for image augmentation.

    Args: Some basic parameters of the image:

        brightness: 
        contrast:
        saturation:
        hue:
    
    Returns:
        ColorJitter image.

    """


    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5) -> None:
        super(RandomColorJitter, self).__init__()
        self.color_jitter = ColorJitter(brightness=brightness, contrast=contrast,
                                        saturation=saturation, hue=hue)

    def forward(self, imgs):
        b, c, h, w = imgs.shape
        imgs = imgs.view(-1, 3, h, w)
        imgs = self.color_jitter(imgs)
        imgs = imgs.view(b, c, h, w)
        return imgs
