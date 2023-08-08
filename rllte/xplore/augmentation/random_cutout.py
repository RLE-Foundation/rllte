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


class RandomCutout(BaseAugmentation):
    """Random Cutout operation for image augmentation.

    Args:
        min_cut (int): Min size of the cut shape.
        max_cut (int): Max size of the cut shape.

    Returns:
        Augmented images.
    """

    def __init__(self, min_cut: int = 10, max_cut: int = 30) -> None:
        super().__init__()
        self.min_cut = min_cut
        self.max_cut = max_cut

    def forward(self, x: th.Tensor) -> th.Tensor:
        n, c, h, w = x.size()
        w1 = th.randint(self.min_cut, self.max_cut, (n,))
        h1 = th.randint(self.min_cut, self.max_cut, (n,))

        cutouts = th.empty((n, c, h, w), dtype=x.dtype, device=x.device)
        for i, (img, w11, h11) in enumerate(zip(x, w1, h1)):
            cut_img = img.clone()
            cut_img[:, h11 : h11 + h11, w11 : w11 + w11] = th.tensor(0, dtype=cut_img.dtype, device=cut_img.device)
            cutouts[i] = cut_img

        return cutouts
