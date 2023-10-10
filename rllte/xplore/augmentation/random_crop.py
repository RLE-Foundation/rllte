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
from torch.nn import functional as F

from rllte.common.prototype import BaseAugmentation


class RandomCrop(BaseAugmentation):
    """Random crop operation for processing image-based observations.

    Args:
        pad (int): Padding size.
        out (int): Desired output size.

    Returns:
        Augmented images.
    """

    def __init__(self, pad: int = 4, out: int = 84) -> None:
        super().__init__()
        self._out = out
        self._pad = pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self._pad] * 4)
        x = F.pad(x, padding, "replicate")

        crop_max = x.size()[2] - self._out + 1
        new_w = th.randint(0, crop_max, (n,))
        new_h = th.randint(0, crop_max, (n,))
        cropped = th.empty(size=(n, c, self._out, self._out), device=x.device)

        for idx, (img, t_h, t_w) in enumerate(zip(x, new_h, new_w)):
            cropped[idx] = img[:, t_h : t_h + self._out, t_w : t_w + self._out]

        return cropped
