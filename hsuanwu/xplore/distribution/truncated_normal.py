from torch.distributions.utils import _standard_normal
import torch

from hsuanwu.common.typing import *
from hsuanwu.xplore.distribution.base import BaseDistribution

class TruncatedNormal(BaseDistribution):
    """Truncated normal distribution for sampling noise.
    
    Args:
        mu: Mean of the distribution.
        sigma: Standard deviation of the distribution.
        low: Lower bound for action range.
        high: Upper bound for action range.
        eps: A constant for clamping.
    
    Returns:
        Truncated normal distribution instance.
    """
    def __init__(self, 
                 mu: Tensor, 
                 sigma: Tensor, 
                 low: float = -1., 
                 high: float = 1., 
                 eps: float = 1e-6) -> None:
        super().__init__(mu, sigma, low, high, eps)


    def sample(self, clip: float = None, sample_shape = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped sample
        
        Args:
            clip: Range for noise truncation operation.
            sample_shape: The size of the sample to be drawn.
        
        Returns:
            A sample_shape shaped sample.
        """
        shape = self._extended_shape(sample_shape)
        noise = _standard_normal(shape,
                               dtype=self._mu.dtype,
                               device=self._mu.device)
        noise *= self._sigma
        if clip is not None:
            # clip the sampled noises
            noise = torch.clamp(noise, -clip, clip)
        x = self._mu + noise
        return self._clamp(x)
