import torch as th
from torch.distributions import Normal

from hsuanwu.xplore.augmentation.base import BaseAugmentation


class GaussianNoise(BaseAugmentation):
    """Gaussian noise operation for processing state-based observations.

    Args:
        mu (float or Tensor): mean of the distribution.
        scale (float or Tensor): standard deviation of the distribution.

    Returns:
        Augmented states.
    """

    def __init__(self, mu: float = 0, sigma: float = 1.0) -> None:
        super().__init__()
        self.dist = Normal(loc=mu, scale=sigma)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 2, "GaussianNoise only supports state-based observations!"
        z = self.dist.sample(x.size())

        return z + x
