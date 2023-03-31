import torch
from torch.distributions.utils import _standard_normal

from hsuanwu.common.typing import Tensor, TorchSize
from hsuanwu.xplore.distribution.base import BaseDistribution


class TruncatedNormalNoise(BaseDistribution):
    """Truncated normal noise.

    Args:
        mu (Tensor): mean of the distribution (often referred to as mu).
        sigma (Tensor): standard deviation of the distribution (often referred to as sigma).
        low (float): Lower bound for action range.
        high (float): Upper bound for action range.
        eps (float): A constant for clamping.

    Returns:
        Truncated normal noise instance.
    """

    def __init__(
        self,
        mu: Tensor,
        sigma: Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(mu, sigma, low, high, eps)

    def sample(self, clip: float = None, sample_shape: TorchSize = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped sample

        Args:
            clip (float): Range for noise truncation operation.
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        shape = self._extended_shape(sample_shape)
        noise = _standard_normal(shape, dtype=self._mu.dtype, device=self._mu.device)
        noise *= self._sigma
        if clip is not None:
            # clip the sampled noises
            noise = torch.clamp(noise, -clip, clip)
        x = self._mu + noise
        return self._clamp(x)
