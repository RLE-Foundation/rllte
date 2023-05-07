import torch as th

from hsuanwu.xplore.augmentation.base import BaseAugmentation


class RandomFlip(BaseAugmentation):
    """Random flip operation for image augmentation.
    Args:
        p (float): The image flip problistily in a batch.

    Returns:
        Augmented images.
    """

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [B, C, H, W]
        bs, channels, h, w = x.size()

        # Flip the images horizontally
        flipped_x = x.flip([3])

        # Generate a random mask to determine which images to flip
        mask = th.rand(bs, device=x.device, dtype=x.dtype) <= self.p
        mask = mask[:, None, None, None]

        # Apply the random flip operation to the input images
        out = mask * flipped_x + (~mask) * x

        return out.view([bs, -1, h, w])
