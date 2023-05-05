import torch as th
import torch.distributions as pyd

from hsuanwu.xplore.distribution import BaseDistribution


class Bernoulli(BaseDistribution):
    """Bernoulli distribution for sampling actions for 'MultiBinary' tasks.
    Args:
        logits (Tensor): The event log probabilities (unnormalized).

    Returns:
        Categorical distribution instance.
    """

    def __init__(
        self,
        logits: th.Tensor,
    ) -> None:
        super().__init__()
        self.dist = pyd.Bernoulli(logits=logits)

    @property
    def probs(self) -> th.Tensor:
        """Return probabilities."""
        return self.dist.probs

    @property
    def logits(self) -> th.Tensor:
        """Returns the unnormalized log probabilities."""
        return self.dist.logits

    def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # noqa B008
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        return self.dist.sample()

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Returns the log of the probability density/mass function evaluated at actions.

        Args:
            actions (Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
        return self.dist.log_prob(actions).sum(-1)

    def entropy(self) -> th.Tensor:
        """Returns the Shannon entropy of distribution."""
        return self.dist.entropy().sum(-1)

    @property
    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        return th.gt(self.dist.probs, 0.5).float()

    @property
    def mean(self) -> th.Tensor:
        """Returns the mean of the distribution."""
        return th.gt(self.dist.probs, 0.5).float()

    @property
    def stddev(self) -> th.Tensor:
        """Returns the standard deviation of the distribution."""
        raise NotImplementedError(f"{self.__class__} does not implement stddev!")

    @property
    def variance(self) -> th.Tensor:
        """Returns the variance of the distribution."""
        raise NotImplementedError(f"{self.__class__} does not implement variance!")

    def reset(self) -> None:
        """Reset the distribution."""
        raise NotImplementedError(f"{self.__class__} does not implement reset!")

    def rsample(self, sample_shape: th.Size = ...) -> th.Tensor:  # B008
        raise NotImplementedError(f"{self.__class__} does not implement rsample!")
