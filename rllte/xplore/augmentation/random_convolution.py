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


class RandomConvolution(BaseAugmentation):
    """Random Convolution operation for image augmentation. Note that imgs should be normalized and torch tensor."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        num_batch, num_stack_channel, img_h, img_w = x.size()
        num_trans = num_batch
        batch_size = int(num_batch / num_trans)

        rand_conv = th.nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(x.device)

        for trans_index in range(num_trans):
            th.nn.init.xavier_normal_(rand_conv.weight.data)
            temp_imgs = x[trans_index * batch_size : (trans_index + 1) * batch_size]
            temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w)  # (batch x stack, channel, h, w)
            rand_out = rand_conv(temp_imgs)
            if trans_index == 0:
                total_out = rand_out
            else:
                total_out = th.cat((total_out, rand_out), 0)
        total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)

        return total_out
