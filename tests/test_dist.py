import pytest
import torch as th

from rllte.xplore.distribution import (
    Bernoulli,
    Categorical,
    DiagonalGaussian,
    MultiCategorical,
    NormalNoise,
    OrnsteinUhlenbeckNoise,
    SquashedNormal,
    TruncatedNormalNoise,
)


@pytest.mark.parametrize(
    "dist_cls",
    [
        Categorical,
        DiagonalGaussian,
        Bernoulli,
        NormalNoise,
        OrnsteinUhlenbeckNoise,
        SquashedNormal,
        TruncatedNormalNoise,
        MultiCategorical,
    ],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_aug(dist_cls, device):
    device = th.device(device)
    batch_size = 3
    action_dim = 7

    if dist_cls in [Categorical, Bernoulli]:
        logits = th.rand(size=(batch_size, action_dim), device=device)
        dist = dist_cls(logits=logits)
        dist.log_prob(dist.sample())
        dist.log_prob(dist.mean)
        dist.entropy()

    if dist_cls in [MultiCategorical]:
        logits = [th.rand(size=(batch_size, action_dim), device=device) for _ in range(4)]
        dist = dist_cls(logits=logits)
        dist.log_prob(dist.sample())
        dist.log_prob(dist.mean)
        dist.entropy()

    if dist_cls in [DiagonalGaussian, SquashedNormal]:
        mu = th.randn(size=(batch_size, action_dim), device=device)
        sigma = th.rand_like(mu)
        dist = dist_cls(loc=mu, scale=sigma)
        dist.log_prob(dist.sample())
        dist.log_prob(dist.mean)

    if dist_cls in [NormalNoise, OrnsteinUhlenbeckNoise, TruncatedNormalNoise]:
        noiseless_action = th.rand(size=(batch_size, action_dim), device=device)
        dist = dist_cls()
        dist.reset(noiseless_action, step=0)
        dist.sample()
        dist.mean

    print("Distribution test passed!")
