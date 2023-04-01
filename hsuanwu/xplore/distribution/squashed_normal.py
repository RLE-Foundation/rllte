import math

import torch
from torch import distributions as pyd
from torch.nn import functional as F

from hsuanwu.common.typing import Tensor, TorchSize
from hsuanwu.xplore.distribution.base import BaseDistribution


class TanhTransform(pyd.transforms.Transform):
    # Borrowed from https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py.
    """Tanh transformation."""

    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

class SquashedNormal(BaseDistribution):
    """Squashed normal distribution for Soft Actor-Critic learner.

    Args:
        mu (Tensor): mean of the distribution (often referred to as mu).
        sigma (Tensor): standard deviation of the distribution (often referred to as sigma).
        low (float): Lower bound for action range.
        high (float): Upper bound for action range.
        eps (float): A constant for clamping.

    Returns:
        Squashed normal distribution instance.
    """

    def __init__(self, mu: Tensor, sigma: Tensor, low: float = -1, high: float = 1, eps: float = 0.000001) -> None:
        super().__init__(mu, sigma, low, high, eps)

        self.tfd = pyd.TransformedDistribution(
            base_distribution=pyd.Normal(loc=mu, scale=sigma),
            transforms=[TanhTransform()]
        )
    
    def sample(self, sample_shape: TorchSize = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.

        Args:
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        return self._clamp(self.tfd.sample(sample_shape))
    
    def rsample(self, sample_shape: TorchSize = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples if the distribution parameters are batched.
        
        Args:
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        return self._clamp(self.tfd.rsample(sample_shape))

    @property
    def mean(self) -> Tensor:
        """Return the transformed mean."""
        mu = self._mu
        for tr in self.tfd.transforms:
            mu = tr(mu)
        return mu
    
    def log_prob(self, value: Tensor) -> Tensor:
        """Scores the sample by inverting the transform(s) and computing the score using the score of the base distribution and the log abs det jacobian."""
        return self.tfd.log_prob(value)