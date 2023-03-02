from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import torch

from hsuanwu.common.typing import *

class TruncatedNormalActionNoise(pyd.Normal):
    """Truncated normal distribution for sampling noise.
    
    Args:
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution.
        low: Lower bound for clipping.
        high: Upper bound for clipping.
        eps: A constant for clamping.
    
    Returns:
        Distribution instance.
    """
    def __init__(self, 
                 loc: Tensor, 
                 scale: Tensor, 
                 low: float = -1.0, 
                 high: float = 1.0, 
                 eps: float = 1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: Tensor) -> Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip: float = None, sample_shape = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            # clip the sampled noises
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
