import torch
import torchvision.transforms as T

from hsuanwu.common.typing import Tensor
from hsuanwu.xplore.augmentation.base import BaseAugmentation


class RandAugment(BaseAugmentation):
    """Random ColorJitter operation for image augmentation.

    Args:
        None.

    Returns:
        Augmented images.
    """

    def __init__(
        self,
    ) -> None:
        super(RandAugment, self).__init__()
        self.auto_augment_function = T.RandAugment()

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()

        # For Channels to split. Like RGB-3 Channels.
        x_list = torch.split(x, 3, dim=1)
        x_aug_list = []
        for x_part in x_list:
            x_part = (x_part*255).clamp(0, 255).to(torch.uint8)
            x_part_aug = self.auto_augment_function(x_part)
            x_aug_list.append(x_part_aug)
        x = torch.cat(x_aug_list, dim=1)
        x = x.float() / 255.0 
        return x
