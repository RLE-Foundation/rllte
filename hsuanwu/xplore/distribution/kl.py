import torch as th
from torch.distributions import register_kl

from hsuanwu.xplore.distribution.categorical import Categorical


@register_kl(Categorical, Categorical)
def kl_categorical_categorical(p, q):
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = th.inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)
