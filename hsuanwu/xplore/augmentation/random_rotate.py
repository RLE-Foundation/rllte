import torch

from hsuanwu.common.typing import *
from hsuanwu.xplore.augmentation.base import BaseAugmentation



class RandomRotate(BaseAugmentation): 
    """Random rotate operation for processing image-based observations.

    Args:
        p: The image rotate problistily in a batch.

    Returns:
        Random rotate image in a batch.
    """
    def __init__(self, 
                 p: float = 0.2) -> None:
        super(RandomRotate, self).__init__()
        self.p = p
        
        
    def forward(self, x: Tensor) -> Tensor:
        # images: [B, C, H, W]
        device = x.device
        bs, channels, h, w = x.size()
        x = x.to(device)

        rot90_images = x.rot90(1, [2, 3])
        rot180_images = x.rot90(2, [2, 3])
        rot270_images = x.rot90(3, [2, 3])

        rnd = torch.rand(size=(bs,), device=device)
        rnd_rot = torch.randint(low=1, high=4, size=(bs,), device=device)
        mask = (rnd <= self.p).float()

        mask = rnd_rot * mask
        mask = mask.long()

        frames = x.shape[1]
        masks = [torch.zeros_like(mask) for _ in range(4)]
        for i, m in enumerate(masks):
            m[mask == i] = 1
            m = m[:, None] * torch.ones([1, frames], device=device).type(mask.dtype).type(x.dtype)
            m = m[:, :, None, None]
            masks[i] = m

        out = masks[0] * x + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images
        out = out.view(bs, -1, h, w)

        return out