import torch as th

from hsuanwu.common.base_augmentation import BaseAugmentation


class RandomTranslate(BaseAugmentation):
    """Random translate operation for processing image-based observations.
    Args:
        size (int): The scale size in translated images
        scale_factor (float): The scale factor ratio in translated images. Should have 0.0 <= scale_factor <= 1.0
    Returns:
        Augmented images.
    """

    def __init__(self, size: int = 256, scale_factor: float = 0.75) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x: th.Tensor) -> th.Tensor:
        # update to support any channels
        _, _, in_h, in_w = x.shape
        x = th.nn.functional.interpolate(
            x,
            size=(int(in_h * self.scale_factor), int(in_w * self.scale_factor)),
            mode="bilinear",
            align_corners=False,
        )
        n, c, h, w = x.shape

        device = x.device
        assert self.size >= h and self.size >= w
        outs = th.zeros((n, c, self.size, self.size), dtype=x.dtype, device=device)
        h1s = th.randint(0, self.size - h + 1, (n,), device=device)
        w1s = th.randint(0, self.size - w + 1, (n,), device=device)
        for out, img, h1, w1 in zip(outs, x, h1s, w1s):
            out[:, h1 : h1 + h, w1 : w1 + w] = img
        return outs
