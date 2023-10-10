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
from torch.distributions import Uniform

from rllte.common.prototype import BaseAugmentation


class RandomAmplitudeScaling(BaseAugmentation):
    """Random amplitude scaling operation for processing state-based observations.

    Args:
        low (float): lower range (inclusive).
        high (float): upper range (exclusive).

    Returns:
        Augmented states.
    """

    def __init__(self, low: float = 0.6, high: float = 1.2) -> None:
        super().__init__()
        self.dist = Uniform(low=low, high=high)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 2, "RandomAmplitudeScaling only supports state-based observations!"
        z = self.dist.sample(x.size())

        return z.to(x.device) * x
