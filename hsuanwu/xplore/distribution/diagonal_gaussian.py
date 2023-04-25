import torch as th
from torch import distributions as pyd

from hsuanwu.xplore.distribution.base import BaseDistribution


class DiagonalGaussian(BaseDistribution):
    """Diagonal Gaussian distribution for 'Box' tasks.

    Args:
        mu (Tensor): The mean of the distribution (often referred to as mu).
        sigma (Tensor): The standard deviation of the distribution (often referred to as sigma).

    Returns:
        Squashed normal distribution instance.
    """

    def __init__(self, mu: th.Tensor, sigma: th.Tensor) -> None:
        super().__init__()

        self._mu = mu
        self._sigma = sigma
        self.dist = pyd.Normal(loc=mu, scale=sigma)

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
        """Return the transformed mean."""
        return self._mu

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Scores the sample by inverting the transform(s) and computing the score using the
            score of the base distribution and the log abs det jacobian.
        Args:
            actions (Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
        return self.dist.log_prob(actions)

    def reset(self) -> None:
        """Reset the distribution."""
        raise NotImplementedError

    def entropy(self) -> th.Tensor:
        """Returns the Shannon entropy of distribution."""
        return self.dist.entropy()

    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        raise NotImplementedError
