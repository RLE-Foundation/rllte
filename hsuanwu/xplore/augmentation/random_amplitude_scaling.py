import torch as th
from torch.distributions import Uniform

from hsuanwu.xplore.augmentation.base import BaseAugmentation


class RandomAmplitudeScaling(BaseAugmentation):
    """Random amplitude scaling operation for processing state-based observations.

    Args:
        low (float): lower range (inclusive).
        high (float): upper range (exclusive).

    Returns:
        Augmented states.
    """

    def __init__(self, low: float = 0.6, high: float = 1.2) -> None:
        super().__init__()
        self.dist = Uniform(low=low, high=high)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert (
            len(x.size()) == 2
        ), "RandomAmplitudeScaling only supports state-based observations!"
        z = self.dist.sample(x.size())

        return z * x
