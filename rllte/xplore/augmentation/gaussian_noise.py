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
from torch.distributions import Normal

from rllte.common.prototype import BaseAugmentation


class GaussianNoise(BaseAugmentation):
    """Gaussian noise operation for processing state-based observations.

    Args:
        mu (float or th.Tensor): mean of the distribution.
        scale (float or th.Tensor): standard deviation of the distribution.

    Returns:
        Augmented states.
    """

    def __init__(self, mu: float = 0, sigma: float = 1.0) -> None:
        super().__init__()
        self.dist = Normal(loc=mu, scale=sigma)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 2, "GaussianNoise only supports state-based observations!"
        z = self.dist.sample(x.size())

        return z.to(x.device) + x
