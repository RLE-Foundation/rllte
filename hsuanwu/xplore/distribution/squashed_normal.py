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
        mu (Tensor): The mean of the distribution (often referred to as mu).
        sigma (Tensor): The standard deviation of the distribution (often referred to as sigma).

    Returns:
        Squashed normal distribution instance.
    """

    def __init__(self, mu: Tensor, sigma: Tensor) -> None:
        super().__init__()

        self._mu = mu
        self._sigma = sigma
        self.dist = pyd.TransformedDistribution(
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
        return self.dist.sample(sample_shape)
    
    def rsample(self, sample_shape: TorchSize = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples if the distribution parameters are batched.
        
        Args:
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        return self.dist.rsample(sample_shape)

    @property
    def mean(self) -> Tensor:
        """Return the transformed mean."""
        mu = self._mu
        for tr in self.dist.transforms:
            mu = tr(mu)
        return mu
    
    def log_prob(self, actions: Tensor) -> Tensor:
        """Scores the sample by inverting the transform(s) and computing the score using the score of the base distribution and the log abs det jacobian.
        Args:
            actions (Tensor): The actions to be evaluated.
        
        Returns:
            The log_prob value.
        """
        return self.dist.log_prob(actions)

    def reset(self) -> None:
        """Reset the distribution.
        """
        raise NotImplementedError

    def entropy(self) -> Tensor:
        """Returns the Shannon entropy of distribution.
        """
        raise NotImplementedError
    
    def mode(self) -> Tensor:
        """Returns the mode of the distribution.
        """
        raise NotImplementedError