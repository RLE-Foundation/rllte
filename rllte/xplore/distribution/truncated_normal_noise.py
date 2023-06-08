import torch as th
import torch.distributions as pyd

from rllte.common.base_distribution import BaseDistribution
from rllte.xplore.distribution import utils


class TruncatedNormalNoise(BaseDistribution):
    """Truncated normal action noise. See Section 3.1 of
        "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning".

    Args:
        loc (float): mean of the noise (often referred to as mu).
        scale (float): standard deviation of the noise (often referred to as sigma).
        stddev_schedule (str): Use the exploration std schedule.
        stddev_clip (float): The exploration std clip range.

    Returns:
        Truncated normal noise instance.
    """

    def __init__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        stddev_schedule: str = "linear(1.0, 0.1, 100000)",
        stddev_clip: float = 0.3,
    ) -> None:
        super().__init__()

        self.loc = loc
        self.scale = scale
        self.dist = pyd.Normal(loc=loc, scale=scale)
        self.noiseless_action = None
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

    def sample(self, clip: bool = False, sample_shape: th.Size = th.Size()) -> th.Tensor:  # noqa B008
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            clip (bool): Whether to perform noise truncation.
            sample_shape (Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        noise = th.as_tensor(
            self.dist.sample(sample_shape=self.noiseless_action.size()),
            device=self.noiseless_action.device,
            dtype=self.noiseless_action.dtype,
        )
        if clip:
            # clip the sampled noises
            noise = th.clamp(noise, -self.stddev_clip, self.stddev_clip)
        return noise + self.noiseless_action

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

    def reset(self, noiseless_action: th.Tensor, step: int = 0) -> None:
        """Reset the noise instance.

        Args:
            noiseless_action (Tensor): Unprocessed actions.
            step (int): Global training step that can be None when there is no noise schedule.

        Returns:
            None.
        """
        self.noiseless_action = noiseless_action
        if self.stddev_schedule is not None:
            # TODO: reset the std of normal distribution.
            self.dist.scale = th.ones_like(self.dist.scale) * utils.schedule(self.stddev_schedule, step)

    @property
    def mean(self) -> th.Tensor:
        """Returns the mean of the distribution."""
        return self.noiseless_action

    @property
    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        return self.noiseless_action

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
