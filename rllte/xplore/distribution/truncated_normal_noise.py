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
import torch.distributions as pyd
from torch.distributions.utils import _standard_normal

from rllte.common.prototype import BaseDistribution
from rllte.xplore.distribution import utils


class TruncatedNormalNoise(BaseDistribution):
    """Truncated normal action noise. See Section 3.1 of
        "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning".

    Args:
        loc (float): mean of the noise (often referred to as mu).
        scale (float): standard deviation of the noise (often referred to as sigma).
        low (float): The lower bound of the noise.
        high (float): The upper bound of the noise.
        eps (float): A small value to avoid numerical instability.
        stddev_schedule (str): Use the exploration std schedule.
        stddev_clip (float): The exploration std clip range.

    Returns:
        Truncated normal noise instance.
    """

    def __init__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
        stddev_schedule: str = "linear(1.0, 0.1, 100000)",
        stddev_clip: float = 0.3,
    ) -> None:
        super().__init__()

        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.eps = eps
        self.dist = pyd.Normal(loc=loc, scale=scale)
        self.noiseless_action = None
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

    def _clamp(self, x: th.Tensor) -> th.Tensor:
        """Clamps the input to the range [low, high]."""
        clamped_x = th.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip: bool = False, sample_shape: th.Size = th.Size()) -> th.Tensor:  # noqa B008
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            clip (bool): Whether to perform noise truncation.
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        noise = _standard_normal(
            self.noiseless_action.size(), dtype=self.noiseless_action.dtype, device=self.noiseless_action.device
        )
        noise *= self.scale

        if clip:
            # clip the sampled noises
            noise = th.clamp(noise, -self.stddev_clip, self.stddev_clip)
        return self._clamp(noise + self.noiseless_action)

    def reset(self, noiseless_action: th.Tensor, step: int = 0) -> None:
        """Reset the noise instance.

        Args:
            noiseless_action (th.Tensor): Unprocessed actions.
            step (int): Global training step that can be None when there is no noise schedule.

        Returns:
            None.
        """
        self.noiseless_action = noiseless_action
        if self.stddev_schedule is not None:
            # TODO: reset the std of normal distribution.
            self.scale = utils.schedule(self.stddev_schedule, step)

    @property
    def mean(self) -> th.Tensor:
        """Returns the mean of the distribution."""
        return self.noiseless_action

    @property
    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        return self.noiseless_action
