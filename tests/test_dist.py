import os
import sys
import pytest

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import torch as th

from rllte.xplore.distribution import (Categorical,
                                       DiagonalGaussian,
                                       Bernoulli,
                                       NormalNoise,
                                       OrnsteinUhlenbeckNoise,
                                       SquashedNormal,
                                       TruncatedNormalNoise)

@pytest.mark.parametrize("dist", [Categorical,
                                  DiagonalGaussian,
                                  Bernoulli,
                                  NormalNoise,
                                  OrnsteinUhlenbeckNoise,
                                  SquashedNormal,
                                  TruncatedNormalNoise])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_aug(dist, device):    
    device = th.device(device)
    if dist in [Categorical, Bernoulli]:
        inputs = th.randn(size=(1, 7), device=device)
        dist = dist(logits=inputs)
        dist.sample()
        dist.mean

        if dist is Categorical:
            dist.log_prob(actions=th.randint(1, 7, size=(1,), device=device))
        if dist is Bernoulli:
            dist.log_prob(actions=th.randint(1, 7, size=(7,), device=device))

    if dist in [DiagonalGaussian, SquashedNormal]:
        mu = th.randn(size=(1, 17), device=device)
        sigma = th.rand(size=(1, 17), device=device)
        dist = dist(loc=mu, scale=sigma)
        dist.sample()
        dist.mean
        dist.log_prob(actions=th.rand(1, 17, device=device))
    
    if dist in [NormalNoise, OrnsteinUhlenbeckNoise, TruncatedNormalNoise]:
        noiseless_action = th.rand(size=(1, 8), device=device)
        dist = dist()
        dist.reset(noiseless_action, step=0)
        dist.sample()
        dist.mean

    print("Distribution test passed!")
