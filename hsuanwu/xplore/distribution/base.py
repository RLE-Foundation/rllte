import torch
from torch import distributions as pyd

from hsuanwu.common.typing import Tensor    

class BaseDistribution(pyd.Normal):
    """Base class of distribution.

    Args:
        mu (Tensor): Mean of the distribution (often referred to as mu).
        sigma (Tensor): Standard deviation of the distribution (often referred to as sigma).

    Returns:
        Base distribution instance.
    """

    def __init__(
        self,
        mu: Tensor,
        sigma: Tensor
    ) -> None:
        super().__init__(loc=mu, scale=sigma, validate_args=False)
        self._mu = mu
        self._sigma = sigma
