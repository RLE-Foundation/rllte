import torch
from torch.distributions.utils import _standard_normal

from hsuanwu.common.typing import Tensor, TorchSize, Optional
from hsuanwu.xplore.distribution.base import BaseDistribution
from hsuanwu.xplore.distribution import utils

class OrnsteinUhlenbeckNoise(BaseDistribution):
    """Ornstein Uhlenbeck action noise.
        Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    Args:
        mu (float): mean of the noise (often referred to as mu).
        sigma (float): standard deviation of the noise (often referred to as sigma).
        theta (float): Rate of mean reversion.
        dt (float): Timestep for the noise.
        initial_noise (float): Initial value for the noise output, (if None: 0)

    Returns:
        Ornstein-Uhlenbeck noise instance.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[Tensor] = None,
        stddev_schedule: str = "linear(1.0, 0.1, 100000)"
    ) -> None:
        super().__init__()

        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._dt = dt
        self._noiseless_action = None
        self._stddev_schedule = stddev_schedule

        self.initial_noise = initial_noise
        self.noise_prev = torch.zeros_like(self._mu)
        self.init()

    def init(self) -> None:
        """Reset the Ornstein Uhlenbeck noise, to the initial position"""
        self.noise_prev = (
            self.initial_noise
            if self.initial_noise is not None
            else torch.zeros_like(self._mu)
        )
    
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
            # TODO: reset the std of 
            self._sigma = utils.schedule(self._stddev_schedule, step)

    def sample(self, clip: float = None, sample_shape: TorchSize = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped sample

        Args:
            clip: Range for noise truncation operation.
            sample_shape: The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * torch.sqrt(self._dt) * _standard_normal(self._noiseless_action.size())
        )
        noise = torch.as_tensor(noise, dtype=self._noiseless_action.dtype, device=self._noiseless_action.device)
        self.noise_prev = noise

        return noise + self._noiseless_action

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the distribution.
        """
        return self._noiseless_action
    
    @property
    def mode(self) -> Tensor:
        """Returns the mode of the distribution.
        """
        return self._noiseless_action
    
    def rsample(self, sample_shape: TorchSize = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.

        Args:
            sample_shape (TorchSize): The size of the sample to be drawn.
        
        Returns:
            A sample_shape shaped sample.
        """
        raise NotImplementedError
    
    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability density/mass function evaluated at `value`.

        Args:
            value (Tensor): The value to be evaluated.
        
        Returns:
            The log_prob value.
        """
        raise NotImplementedError

    def entropy(self) -> Tensor:
        """Returns the Shannon entropy of distribution.
        """
        raise NotImplementedError
