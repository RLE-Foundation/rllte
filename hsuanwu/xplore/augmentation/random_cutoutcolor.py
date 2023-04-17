import torch

from hsuanwu.common.typing import Tensor
from hsuanwu.xplore.augmentation.base import BaseAugmentation


class RandomCutoutColor(BaseAugmentation):
    """Random Cutout operation for image augmentation.
    Args: the size of the cut area
        min_cut: min size of the cut shape.
        max_cut: max size of the cut shape.

    Returns:
        Augmented images.
    """

    def __init__(self, min_cut: int = 10, max_cut: int = 30) -> None:
        super(RandomCutoutColor, self).__init__()
        self.min_cut = min_cut
        self.max_cut = max_cut

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.size()

        w1 = torch.randint(self.min_cut, self.max_cut, (n,))
        h1 = torch.randint(self.min_cut, self.max_cut, (n,))

        cutouts = torch.empty((n, c, h, w), dtype=x.dtype, device=x.device)
        rand_box = torch.rand((n, c), device=x.device)

        for i, (img, w11, h11) in enumerate(zip(x, w1, h1)):
            cut_img = img.clone()

            rand_color = (
                rand_box[i]
                .reshape(-1, 1, 1)
                .expand_as(cut_img[:, h11 : h11 + h11, w11 : w11 + w11])
            )
            cut_img[:, h11 : h11 + h11, w11 : w11 + w11] = rand_color

            cutouts[i] = cut_img

        return cutouts
