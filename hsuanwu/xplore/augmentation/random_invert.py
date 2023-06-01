import torch as th
import torchvision.transforms as T

from hsuanwu.common.base_augmentation import BaseAugmentation


class RandomInvert(BaseAugmentation):
    """RandomInvert method based on “RandomInvert: Inverts the colors of the given image randomly with a given probability”.
    Args:
        p (float) : probability of the image being color inverted. Default value is 0.5
    Returns:
        Augmented images.
    """

    def __init__(
        self,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.p = p
        self.augment_function = T.RandomInvert(p=self.p)

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
