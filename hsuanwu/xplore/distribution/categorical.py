import torch.distributions as pyd

from hsuanwu.common.typing import *


class Categorical(pyd.Categorical):
    """Categorical distribution for sampling actions in discrete control tasks.
    Args:
        logits: event log probabilities (unnormalized).

    Returns:
        Categorical distribution instance.
    """

    def __init__(self, logits=None):
        super().__init__(probs=None, logits=logits, validate_args=None)

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    @property
    def mean(self) -> Tensor:
        return self.probs.argmax(dim=-1, keepdim=True)
