import torch
from torchvision.transforms import ColorJitter

from hsuanwu.common.typing import Tensor
from hsuanwu.xplore.augmentation.base import BaseAugmentation


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

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.5,
    ) -> None:
        super(RandomColorJitter, self).__init__()
        self.color_jitter = ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()

        # For Channels to split. Like RGB-3 Channels.
        x_list = torch.split(x, 3, dim=1)
        x_aug_list = []
        for x_part in x_list:
            x_part_aug = self.color_jitter(x_part)
            x_aug_list.append(x_part_aug)

        x = torch.cat(x_aug_list, dim=1)

        return x
