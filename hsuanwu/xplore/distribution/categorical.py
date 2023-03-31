import torch.distributions as pyd

from hsuanwu.common.typing import Tensor, TorchSize
from hsuanwu.xplore.distribution import BaseDistribution


class Categorical(BaseDistribution):
    """Categorical distribution for sampling actions in discrete control tasks.
    Args:
        mu (Tensor): mean of the distribution. For Categorical distribution, mu denotes the "logits": event log probabilities (unnormalized).
        sigma (Tensor): Deprecated here.
        low (float): Deprecated here.
        high (float): Deprecated here.
        eps (float): Deprecated here.

    Returns:
        Categorical distribution instance.
    """

    def __init__(
        self,
        mu: Tensor,
        sigma: None,
        low: None,
        high: None,
        eps: None,
    ) -> None:
        super().__init__(mu, sigma, low, high, eps)
        self.dist = pyd.Categorical(logits=mu)

    def sample(self):
        return self.dist.sample().unsqueeze(-1)

    def log_probs(self, actions) -> Tensor:
        return (
            self.dist
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    @property
    def mean(self) -> Tensor:
        return self.dist.probs.argmax(dim=-1, keepdim=True)
