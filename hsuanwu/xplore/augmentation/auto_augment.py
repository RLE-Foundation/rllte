import torch as th
import torchvision.transforms as T

from hsuanwu.xplore.augmentation.base import BaseAugmentation


class AutoAugment(BaseAugmentation):
    """Augmentation method based on “AutoAugment: Learning Augmentation Strategies from Data”.
    Args:
        augment_policy (str): Desired policy enum defined by torchvision.transforms.autoaugment.AutoAugmentPolicy. Default is AutoAugmentPolicy.IMAGENET.
    Returns:
        Augmented images.
    """

    def __init__(
        self,
        augment_policy: str = T.AutoAugmentPolicy.IMAGENET,
    ) -> None:
        super(AutoAugment, self).__init__()
        self.policy = augment_policy
        self.auto_augment_function = T.AutoAugment(self.policy)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, h, w = x.size()

        # For Channels to split. Like RGB-3 Channels.
        x_list = th.split(x, 3, dim=1)
        x_aug_list = []
        for x_part in x_list:
            x_part = (x_part * 255).clamp(0, 255).to(th.uint8)
            x_part_aug = self.auto_augment_function(x_part)
            x_aug_list.append(x_part_aug)
        x = th.cat(x_aug_list, dim=1)
        x = x.float() / 255.0
        return x
