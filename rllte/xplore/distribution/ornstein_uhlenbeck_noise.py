import numpy as np
import torch as th
from torch.distributions.utils import _standard_normal

from rllte.common.base_distribution import BaseDistribution
from rllte.xplore.distribution import utils


class OrnsteinUhlenbeckNoise(BaseDistribution):
    """Ornstein Uhlenbeck action noise.
        Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    Args:
        loc (float): mean of the noise (often referred to as mu).
        scale (float): standard deviation of the noise (often referred to as sigma).
        low (float): The lower bound of the noise.
        high (float): The upper bound of the noise.
        eps (float): A small value to avoid numerical instability.
        theta (float): The rate of mean reversion.
        dt (float): Timestep for the noise.
        stddev_schedule (str): Use the exploration std schedule.
        stddev_clip (float): The exploration std clip range.

    Returns:
        Ornstein-Uhlenbeck noise instance.
    """

    def __init__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        low: float = -1.0, 
        high: float = 1.0, 
        eps: float = 1e-6,
        theta: float = 0.15,
        dt: float = 1e-2,
        stddev_schedule: str = "linear(1.0, 0.1, 100000)",
        stddev_clip: float = 0.3,
    ) -> None:
        super().__init__()

        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.eps = eps
        self.theta = theta
        self.dt = dt
        self.noiseless_action = None
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.noise_prev = None

    def reset(self, noiseless_action: th.Tensor, step: int = 0) -> None:
        """Reset the noise instance.

        Args:
            noiseless_action (Tensor): Unprocessed actions.
            step (int): Global training step that can be None when there is no noise schedule.

        Returns:
            None.
        """
        self.noiseless_action = noiseless_action
        if self.noise_prev is None:
            self.noise_prev = th.zeros_like(self.noiseless_action)
        if self.stddev_schedule is not None:
            # TODO: reset the std of
            self.scale = utils.schedule(self.stddev_schedule, step)
    
    def _clamp(self, x: th.Tensor) -> th.Tensor:
        """Clamps the input to the range [low, high].
        """
        clamped_x = th.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip: bool = False, sample_shape: th.Size = th.Size()) -> th.Tensor:  # noqa B008
        """Generates a sample_shape shaped sample

        Args:
            clip (bool): Range for noise truncation operation.
            sample_shape (Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        noise = (
            self.noise_prev
            + self.theta * (self.loc - self.noise_prev) * self.dt
            + self.scale
            * np.sqrt(self.dt)
            * _standard_normal(
                self.noiseless_action.size(),
                dtype=self.noiseless_action.dtype,
                device=self.noiseless_action.device,
            )
        )
        noise = th.as_tensor(
            noise,
            dtype=self.noiseless_action.dtype,
            device=self.noiseless_action.device,
        )
        self.noise_prev = noise

        if clip:
            # clip the sampled noises
            noise = th.clamp(noise, -self.stddev_clip, self.stddev_clip)

        return self._clamp(noise + self.noiseless_action)

    @property
    def mean(self) -> th.Tensor:
        """Returns the mean of the distribution."""
        return self.noiseless_action

    @property
    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        return self.noiseless_action

    def rsample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # noqa B008
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            sample_shape (Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        raise NotImplementedError(f"{self.__class__} does not implement rsample!")

    def log_prob(self, value: th.Tensor) -> th.Tensor:
        """Returns the log of the probability density/mass function evaluated at `value`.

        Args:
            value (Tensor): The value to be evaluated.

        Returns:
            The log_prob value.
        """
        raise NotImplementedError(f"{self.__class__} does not implement log_prob!")

    def entropy(self) -> th.Tensor:
        """Returns the Shannon entropy of distribution."""
        raise NotImplementedError(f"{self.__class__} does not implement entropy!")

    @property
    def stddev(self) -> th.Tensor:
        """Returns the standard deviation of the distribution."""
        raise NotImplementedError(f"{self.__class__} does not implement stddev!")

    @property
    def variance(self) -> th.Tensor:
        """Returns the variance of the distribution."""
        raise NotImplementedError(f"{self.__class__} does not implement variance!")
