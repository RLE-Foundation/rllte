import torch
from torch.distributions.utils import _standard_normal

from hsuanwu.common.typing import *
from hsuanwu.xplore.distribution.base import BaseDistribution


class OrnsteinUhlenbeckNoise(BaseDistribution):
    """Ornstein Uhlenbeck action noise.
        Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    Args:
        mu: Mean of the distribution.
        sigma: Standard deviation of the distribution.
        low: Lower bound for action range.
        high: Upper bound for action range.
        eps: A constant for clamping.
        theta: Rate of mean reversion.
        dt: Timestep for the noise.
        initial_noise: Initial value for the noise output, (if None: 0)

    Returns:
        Ornstein-Uhlenbeck noise instance.
    """

    def __init__(
        self,
        mu: Tensor,
        sigma: Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[Tensor] = None,
    ) -> None:
        super().__init__(mu, sigma, low, high, eps)

        self._theta = theta
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = torch.zeros_like(self._mu)
        self.reset()

    def reset(self) -> None:
        """Reset the Ornstein Uhlenbeck noise, to the initial position"""
        self.noise_prev = (
            self.initial_noise
            if self.initial_noise is not None
            else torch.zeros_like(self._mu)
        )

    def sample(self, clip: float = None, sample_shape=torch.Size()) -> Tensor:
        """Generates a sample_shape shaped sample

        Args:
            clip: Range for noise truncation operation.
            sample_shape: The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        shape = self._extended_shape(sample_shape)
        noise = _standard_normal(shape, dtype=self._mu.dtype, device=self._mu.device)

        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * torch.sqrt(self._dt) * noise
        )
        self.noise_prev = noise

        if clip is not None:
            # clip the sampled noises
            noise = torch.clamp(noise, -clip, clip)
        x = self._mu + noise
        return self._clamp(x)
