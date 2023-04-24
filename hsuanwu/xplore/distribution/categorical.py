import torch as th
import torch.distributions as pyd

from hsuanwu.xplore.distribution import BaseDistribution


class Categorical(BaseDistribution):
    """Categorical distribution for sampling actions in discrete control tasks.
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
        self._logits = logits
        self.dist = pyd.Categorical(logits=logits)

    @property
    def probs(self) -> th.Tensor:
        """Return probabilities."""
        return self.dist.probs

    @property
    def logits(self) -> th.Tensor:
        """Returns the unnormalized log probabilities."""
        return self.dist.logits

    def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.

        Args:
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        return self.dist.sample().unsqueeze(-1)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Returns the log of the probability density/mass function evaluated at `value`.

        Args:
            actions (Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
        return self.dist.log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self) -> th.Tensor:
        """Returns the Shannon entropy of distribution."""
        return self.dist.entropy()

    @property
    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        return self.dist.probs.argmax(dim=-1, keepdim=True)

    @property
    def mean(self) -> th.Tensor:
        """Returns the mean of the distribution."""
        return self.dist.probs.argmax(dim=-1, keepdim=True)

    def reset(self) -> None:
        """Reset the distribution."""
        raise NotImplementedError

    def rsample(self, sample_shape: th.Size = ...) -> th.Tensor:
        raise NotImplementedError
