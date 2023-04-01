import torch
from torch.distributions.utils import _standard_normal

from hsuanwu.common.typing import Tensor, TorchSize
from hsuanwu.xplore.distribution.base import BaseDistribution
from hsuanwu.xplore.distribution import utils


class NormalNoise(BaseDistribution):
    """Gaussian action noise.

    Args:
        mu (float): Mean of the noise (often referred to as mu).
        sigma (float): Standard deviation of the noise (often referred to as sigma).
        stddev_schedule (str): Use the exploration std schedule.
        stddev_clip (float): The exploration std clip range.

    Returns:
        Gaussian noise instance.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        stddev_schedule: str = "linear(1.0, 0.1, 100000)"
    ) -> None:
        super().__init__(mu, sigma)

        self._noiseless_action = None
        self._stddev_schedule = stddev_schedule
    
    def reset(self, noiseless_action: Tensor, step: int = None) -> None:
        """Reset the noise instance.
        
        Args:
            noiseless_action (Tensor): Unprocessed actions.
            step (int): Global training step that can be None when there is no noise schedule.
        
        Returns:
            None.
        """
        self._noiseless_action = noiseless_action
        if self._stddev_schedule is not None:
            # TODO: reset the std of normal distribution.
            self.scale = utils.schedule(self._stddev_schedule, step)

    def sample(self, clip: bool = False, sample_shape: TorchSize = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.

        Args:
            clip (bool): Whether to perform noise truncation.
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        noise = torch.as_tensor(super().sample(sample_shape=self._noiseless_action.size()), 
                                device=self._noiseless_action.device, 
                                dtype=self._noiseless_action.dtype)

        return noise + self._noiseless_action

    @property
    def mean(self):
        return self._noiseless_action
