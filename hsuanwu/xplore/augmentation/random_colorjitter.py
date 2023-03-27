

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

# TODO From torch original version
# class RandomColorJitter( BaseAugmentation):

#     def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5, p=0.5, batch_size=128):
#         super(RandomColorJitter, self).__init__()
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         self.hue = hue
#         self.p = p
#         self.batch_size = batch_size
        
#     def forward(self, x):
#         if self.training:
#             x = self.color_jitter(x, self.brightness, self.contrast, self.saturation, self.hue, self.p, self.batch_size)
#         return x
    
#     @staticmethod
#     def color_jitter(x, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5, p=0.5, batch_size=128):
#         b, c, h, w = x.size()

#         # brightness jitter
#         if brightness > 0 and torch.rand(1) < p:
#             brightness_factor = torch.clamp(torch.randn(b, 1, 1, 1) * brightness + 1, 0, 2)
#             x = x * brightness_factor.to(x.dtype).to(x.device)

#         # contrast jitter
#         if contrast > 0 and torch.rand(1) < p:
#             contrast_factor = torch.clamp(torch.randn(b, 1, 1, 1) * contrast + 1, 0)
#             mean = torch.mean(x, axis=(2, 3), keepdim=True)
#             x = (x - mean) * contrast_factor.to(x.dtype).to(x.device) + mean

#         # saturation jitter
#         if saturation > 0 and torch.rand(1) < p:
#             saturation_factor = torch.clamp(torch.randn(b, 1, 1, 1) * saturation + 1, 0)
#             x_hsv = torch.nn.functional.rgb_to_hsv(x)
#             x_hsv[..., 1] *= saturation_factor.to(x.dtype).to(x.device)
#             x = torch.nn.functional.hsv_to_rgb(x_hsv)

#         # hue jitter
#         if hue > 0 and torch.rand(1) < p:
#             hue_factor = torch.randn(b, 1, 1, 1) * hue
#             x_hsv = torch.nn.functional.rgb_to_hsv(x)
#             x_hsv[..., 0] += hue_factor.to(x.dtype).to(x.device)
#             x_hsv[..., 0] -= (x_hsv[..., 0] > 1.0).to(x_hsv.dtype).to(x_hsv.device) * 1.0
#             x_hsv[..., 0] += (x_hsv[..., 0] < 0.0).to(x_hsv.dtype).to(x_hsv.device) * 1.0
#             x = torch.nn.functional.hsv_to_rgb(x_hsv)

#         return x

