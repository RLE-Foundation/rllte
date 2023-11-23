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
import torch.nn.functional as F
from rllte.common.prototype import BaseAugmentation
from .pad_crop import PadCrop
from .pad_resize import PadResizePlus

class PeriodicPadCropResize(BaseAugmentation):
    """
    Periodically applies PadCrop and PadResizePlus transformations to images.

    Args:
        T (int): Number of cycles to apply the transformations. Each cycle consists of one application of PadCrop followed by one application of PadResizePlus.
        pad (int): The padding size.
        highest_pad_strength (int): The maximum strength of padding.

    Note:
        - The PadCrop instance should be initialized with the desired padding size. This size determines how much the images will be padded (and subsequently shifted) in each cycle.
        - The PadResizePlus instance should be initialized with the highest padding strength. This strength determines the range of padding variability that can be applied during the resize and crop operations.
    """

    def __init__(self, pad: int, highest_pad_strength: int,T: int) -> None:
        """
        Initializes the PeriodicPadCropResize module with specified instances of PadCrop and PadResizePlus and the number of cycles.

        Args:
            pad_crop (PadCrop): An initialized instance of the PadCrop class. It applies padding and random shifts to the images.
            pad_resize_plus (PadResizePlus): An initialized instance of the PadResizePlus class. It applies padding, cropping, and resizing to the images.
            T (int): The number of cycles for applying the transformations. In each cycle, PadCrop is applied first, followed by PadResizePlus.
        """
        super().__init__()
        self.pad_crop = PadCrop(pad)
        self.pad_resize_plus = PadResizePlus(highest_pad_strength)
        self.T = T

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Applies the PadCrop and PadResizePlus transformations periodically to the input images.

        Args:
            x (th.Tensor): The input images to be transformed.

        Returns:
            th.Tensor: The transformed images after applying the periodic transformations.
        """
        for _ in range(self.T):
            x = self.pad_crop(x)
            x = self.pad_resize_plus(x)
        return x

