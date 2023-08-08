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
