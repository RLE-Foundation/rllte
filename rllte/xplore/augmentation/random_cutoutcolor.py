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


class RandomCutoutColor(BaseAugmentation):
    """Random Cutout operation for image augmentation.

    Args: the size of the cut area
        min_cut (int): min size of the cut shape.
        max_cut (int): max size of the cut shape.

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
        rand_box = th.rand((n, c), device=x.device)

        for i, (img, w11, h11) in enumerate(zip(x, w1, h1)):
            cut_img = img.clone()

            rand_color = rand_box[i].reshape(-1, 1, 1).expand_as(cut_img[:, h11 : h11 + h11, w11 : w11 + w11])
            cut_img[:, h11 : h11 + h11, w11 : w11 + w11] = rand_color

            cutouts[i] = cut_img

        return cutouts
