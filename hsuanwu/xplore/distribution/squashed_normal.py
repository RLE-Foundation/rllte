from torch.nn import functional as F
from torch import distributions as pyd
import torch
import math

from hsuanwu.common.typing import *


class TanhTransform(pyd.transforms.Transform):
    """Tanh transformation. 
    """
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
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))
    

class SquashedNormal(pyd.TransformedDistribution):
    """Squashed normal distribution for Soft Actor-Critic.
    
    Args:
        mu: Mean of the distribution.
        sigma: Standard deviation of the distribution.
        low: Lower bound for action range.
        high: Upper bound for action range.
        eps: A constant for clamping.
    
    Returns:
        Squashed normal distribution instance.
    """
    def __init__(self, 
                 mu: Tensor, 
                 sigma: Tensor, 
                 low: float = -1., 
                 high: float = 1., 
                 eps: float = 1e-6) -> None:
        self._mu = mu
        self._sigma = sigma
        self._low = low
        self._high = high
        self._eps = eps

        self._base_dist = pyd.Normal(mu, sigma)
        transforms = [TanhTransform()]
        super().__init__(self._base_dist, transforms)


    def _clamp(self, x: Tensor) -> Tensor:
        """ Clamping operation.
        Args:
            x: Tensor to be clamped.
        
        Returns:
            Clamped tensor.
        """
        clamped_x = torch.clamp(
            x, self._low + self._eps, self._high - self._eps)
        x = x - x.detach() + clamped_x.detach()
        return x
    

    def sample(self, sample_shape=torch.Size()):
        """Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.
        """
        return self._clamp(super().sample(sample_shape))


    def rsample(self, sample_shape=torch.Size()):
        """Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples if the distribution parameters are batched.
        """
        return self._clamp(super().rsample(sample_shape))


    @property
    def mean(self):
        """Return the transformed mean.
        """
        mu = self._mu
        for tr in self.transforms:
            mu = tr(mu)
        return mu
