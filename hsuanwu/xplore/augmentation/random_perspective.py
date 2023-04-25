import torch as th
import torchvision.transforms as T

from hsuanwu.xplore.augmentation.base import BaseAugmentation


class RandomPerspective(BaseAugmentation):
    """RandomPerspective method based on “RandomPerspective: Performs
        a random perspective transformation of the given image with a given probability.”.
    Args:
        distortion_scale (float) : argument to control the degree of distortion and ranges from 0 to 1. Default is 0.5.
        p (float) : Smoothness of displacements. Default is 5.0.
        interpolation (Union, InterpolationMode) : Desired interpolation enum defined by
            torchvision.transforms.InterpolationMode. Default is InterpolationMode.BILINEAR.
        fill (sequence or int number) : Pixel fill value for the area outside the transformed image. Default is 0.
    Returns:
        Augmented images.
    """

    def __init__(
        self,
        distortion_scale: float = 0.5,
        p: float = 0.5,
        interpolation: int = 0,
        fill=0,
    ) -> None:
        super().__init__()
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = interpolation
        self.fill = fill

        self.augment_function = T.RandomPerspective(
            distortion_scale=self.distortion_scale,
            p=self.p,
            interpolation=self.interpolation,
            fill=self.fill,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, h, w = x.size()
        # For Channels to split. Like RGB-3 Channels.
        x_list = th.split(x, 3, dim=1)
        x_aug_list = []
        for x_part in x_list:
            x_part = (x_part * 255).clamp(0, 255).to(th.uint8)
            x_part_aug = self.augment_function(x_part)
            x_aug_list.append(x_part_aug)
        x = th.cat(x_aug_list, dim=1)
        x = x.float() / 255.0
        return x
