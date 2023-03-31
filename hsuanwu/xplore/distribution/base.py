import torch
from torch import distributions as pyd

from hsuanwu.common.typing import Tensor, TorchSize    

class BaseDistribution(pyd.Normal):
    """Base class of distribution.

    Args:
        mu (Tensor): mean of the distribution (often referred to as mu).
        sigma (Tensor): standard deviation of the distribution (often referred to as sigma).
        low (float): Lower bound for action range.
        high (float): Upper bound for action range.
        eps (float): A constant for clamping.

    Returns:
        Base distribution instance.
    """

    def __init__(
        self,
        mu: Tensor,
        sigma: Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(loc=mu, scale=sigma, validate_args=False)
        self._mu = mu
        self._sigma = sigma
        self._low = low
        self._high = high
        self._eps = eps

    def _clamp(self, x: Tensor) -> Tensor:
        """Clamping operation for sampled actions.
        Args:
            x: Tensor to be clamped.

        Returns:
            Clamped tensor.
        """
        clamped_x = torch.clamp(x, self._low + self._eps, self._high - self._eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, sample_shape: TorchSize = torch.Size()):
        """Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.

        Args:
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        pass
