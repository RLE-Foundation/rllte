# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


import torch as th

from rllte.common.prototype import BaseAugmentation


class RandomRotate(BaseAugmentation):
    """Random rotate operation for processing image-based observations.

    Args:
        p (float): The image rotate problistily in a batch.

    Returns:
        Random rotate image in a batch.
    """

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: th.Tensor) -> th.Tensor:
        # images: [B, C, H, W]
        device = x.device
        bs, channels, h, w = x.size()
        x = x.to(device)

        rot90_images = x.rot90(1, [2, 3])
        rot180_images = x.rot90(2, [2, 3])
        rot270_images = x.rot90(3, [2, 3])

        rnd = th.rand(size=(bs,), device=device)
        rnd_rot = th.randint(low=1, high=4, size=(bs,), device=device)
        mask = (rnd <= self.p).float()

        mask = rnd_rot * mask
        mask = mask.long()

        frames = x.shape[1]
        masks = [th.zeros_like(mask) for _ in range(4)]
        for i, m in enumerate(masks):
            m[mask == i] = 1
            m = m[:, None] * th.ones([1, frames], device=device).type(mask.dtype).type(x.dtype)
            m = m[:, :, None, None]
            masks[i] = m

        out = masks[0] * x + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images
        out = out.view(bs, -1, h, w)

        return out
