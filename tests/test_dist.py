import pytest
import torch as th

from rllte.xplore.distribution import (
    Bernoulli,
    Categorical,
    DiagonalGaussian,
    NormalNoise,
    OrnsteinUhlenbeckNoise,
    SquashedNormal,
    TruncatedNormalNoise,
)


@pytest.mark.parametrize(
    "dist_cls",
    [Categorical, DiagonalGaussian, Bernoulli, NormalNoise, OrnsteinUhlenbeckNoise, SquashedNormal, TruncatedNormalNoise],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_aug(dist_cls, device):
    device = th.device(device)
    if dist_cls in [Categorical, Bernoulli]:
        inputs = th.rand(size=(1, 7), device=device)
        dist = dist_cls(logits=inputs)
        dist.sample()
        print(dist.mean)

        if dist_cls is Categorical:
            dist.log_prob(actions=th.randint(1, 7, size=(1,), device=device).float())
        if dist_cls is Bernoulli:
            dist.log_prob(actions=th.randint(0, 2, size=(7,), device=device).float())

    if dist_cls in [DiagonalGaussian, SquashedNormal]:
        mu = th.randn(size=(1, 17), device=device)
        sigma = th.rand(size=(1, 17), device=device)
        dist = dist_cls(loc=mu, scale=sigma)
        dist.sample()
        print(dist.mean)
        dist.log_prob(actions=th.rand(1, 17, device=device))

    if dist_cls in [NormalNoise, OrnsteinUhlenbeckNoise, TruncatedNormalNoise]:
        noiseless_action = th.rand(size=(1, 8), device=device)
        dist = dist_cls()
        dist.reset(noiseless_action, step=0)
        dist.sample()
        print(dist.mean)

    print("Distribution test passed!")
