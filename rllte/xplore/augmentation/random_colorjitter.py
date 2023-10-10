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
from torchvision.transforms import ColorJitter

from rllte.common.prototype import BaseAugmentation


class RandomColorJitter(BaseAugmentation):
    """Random ColorJitter operation for image augmentation.

    Args:
        brightness (float): How much to jitter brightness. Should be non negative numbers.
        contrast (float): How much to jitter contrast. Should be non negative numbers.
        saturation (float): How much to jitter saturation. Should be non negative numbers.
        hue (float): How much to jitter hue. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

    Returns:
        Augmented images.
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.5,
    ) -> None:
        super().__init__()
        self.color_jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, h, w = x.size()

        # For Channels to split. Like RGB-3 Channels.
        x_list = th.split(x, 3, dim=1)
        x_aug_list = []
        for x_part in x_list:
            x_part_aug = self.color_jitter(x_part)
            x_aug_list.append(x_part_aug)

        x = th.cat(x_aug_list, dim=1)

        return x
