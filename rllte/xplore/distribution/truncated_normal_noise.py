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


from typing import Optional, TypeVar, Union

import torch as th
from torch.distributions.utils import _standard_normal

from rllte.common.prototype import BaseDistribution
from rllte.common.utils import schedule

SelfTruncatedNormalNoise = TypeVar("SelfTruncatedNormalNoise", bound="TruncatedNormalNoise")


class TruncatedNormalNoise(BaseDistribution):
    """Truncated normal action noise. See Section 3.1 of
        "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning".

    Args:
        mu (Union[float, th.Tensor]): Mean of the noise.
        sigma (Union[float, th.Tensor]): Standard deviation of the noise.
        low (float): The lower bound of the noise.
        high (float): The upper bound of the noise.
        eps (float): A small value to avoid numerical instability.
        stddev_schedule (str): Use the exploration std schedule, available options are:
            `linear(init, final, duration)` and `step_linear(init, final1, duration1, final2, duration2)`.

    Returns:
        Truncated normal noise instance.
    """

    def __init__(
        self,
        mu: Union[float, th.Tensor] = 0.0,
        sigma: Union[float, th.Tensor] = 1.0,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
        stddev_schedule: str = "linear(1.0, 0.1, 100000)",
    ) -> None:
        super().__init__()

        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high
        self.eps = eps
        self.stddev_schedule = stddev_schedule
        self.step = 0

    def __call__(self: SelfTruncatedNormalNoise, noiseless_action: th.Tensor) -> SelfTruncatedNormalNoise:
        """Create the action noise.

        Args:
            noiseless_action (th.Tensor): Unprocessed actions.

        Returns:
            Truncated normal noise instance.
        """
        self.noiseless_action = noiseless_action
        self.scale = schedule(self.stddev_schedule, self.step)
        return self

    def _clamp(self, x: th.Tensor) -> th.Tensor:
        """Clamps the input to the range [low, high]."""
        clamped_x = th.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip: Optional[float] = None, sample_shape: th.Size = th.Size()) -> th.Tensor:  # type: ignore[override]
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            clip (Optional[float]): The clip range of the sampled noises.
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        noise = _standard_normal(
            self.noiseless_action.size(), dtype=self.noiseless_action.dtype, device=self.noiseless_action.device
        )
        noise *= self.scale

        if clip is not None:
            # clip the sampled noises
            noise = th.clamp(noise, -clip, clip)

        self.step += 1

        return self._clamp(noise + self.noiseless_action)

    @property
    def mean(self) -> th.Tensor:
        """Returns the mean of the distribution."""
        return self.noiseless_action

    @property
    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        return self.noiseless_action
