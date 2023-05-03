import torch as th
from torch import distributions as pyd

from hsuanwu.xplore.distribution.base import BaseDistribution


class DiagonalGaussian(BaseDistribution):
    """Diagonal Gaussian distribution for 'Box' tasks.

    Args:
        loc (Tensor): The mean of the distribution (often referred to as mu).
        scale (Tensor): The standard deviation of the distribution (often referred to as sigma).

    Returns:
        Squashed normal distribution instance.
    """

    def __init__(self, loc: th.Tensor, scale: th.Tensor) -> None:
        super().__init__()

        self.loc = loc
        self.scale = scale
        self.dist = pyd.Normal(loc=loc, scale=scale)

    def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # noqa B008
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            sample_shape (Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        return self.dist.sample(sample_shape)

    def rsample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # noqa B008
        """Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of
            reparameterized samples if the distribution parameters are batched.

        Args:
            sample_shape (Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        return self.dist.rsample(sample_shape)

    @property
    def mean(self) -> th.Tensor:
        """Returns the mean of the distribution."""
        return self.loc
    
    @property
    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        return self.loc
    
    @property
    def stddev(self) -> th.Tensor:
        """Returns the standard deviation of the distribution."""
        raise self.scale
    
    @property
    def variance(self) -> th.Tensor:
        """Returns the variance of the distribution."""
        return self.stddev.pow(2)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Returns the log of the probability density/mass function evaluated at actions.

        Args:
            actions (Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
        return self.dist.log_prob(actions).sum(-1)

    def reset(self) -> None:
        """Reset the distribution."""
        raise NotImplementedError(f"{self.__class__} does not implement reset!")

    def entropy(self) -> th.Tensor:
        """Returns the Shannon entropy of distribution."""
        return self.dist.entropy()
