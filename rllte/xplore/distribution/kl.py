import torch as th
from torch.distributions import register_kl

from rllte.xplore.distribution.categorical import Categorical
from rllte.xplore.distribution.diagonal_gaussian import DiagonalGaussian


@register_kl(Categorical, Categorical)
def kl_categorical_categorical(p, q):
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = th.inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)


@register_kl(DiagonalGaussian, DiagonalGaussian)
def kl_diagonal_gaussian_diagonal_gaussian(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
