# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


import torch as th
from torch.distributions import register_kl

from .bernoulli import Bernoulli
from .categorical import Categorical
from .diagonal_gaussian import DiagonalGaussian


@register_kl(Bernoulli, Bernoulli)
def kl_bernoulli_bernoulli(p, q):
    t1 = p.probs * (th.nn.functional.softplus(-q.logits) - th.nn.functional.softplus(-p.logits))
    t1[q.probs == 0] = th.inf
    t1[p.probs == 0] = 0
    t2 = (1 - p.probs) * (th.nn.functional.softplus(q.logits) - th.nn.functional.softplus(p.logits))
    t2[q.probs == 1] = th.inf
    t2[p.probs == 1] = 0
    return t1 + t2


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
