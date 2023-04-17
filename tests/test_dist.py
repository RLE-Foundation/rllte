import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import torch

from hsuanwu.xplore.distribution import (
    Categorical,
    NormalNoise,
    OrnsteinUhlenbeckNoise,
    SquashedNormal,
    TruncatedNormalNoise,
)

if __name__ == "__main__":
    device = torch.device("cuda:0")

    dist = SquashedNormal(mu=torch.rand(1, 17), sigma=torch.rand(1, 17))

    print(dist.sample())
    print(dist.rsample())
    print(dist.log_prob(actions=torch.rand(1, 17)))

    dist = Categorical(logits=torch.randn(1, 7))
    print(dist.sample())
    print(dist.mode)
    print(dist.log_prob(actions=torch.randint(1, 7, size=(2,))))

    dist = TruncatedNormalNoise()
    noiseless_action = torch.rand(1, 8)
    dist.reset(noiseless_action, step=0)
    print(dist.sample())
    print(noiseless_action, dist.mean)

    dist = NormalNoise()
    noiseless_action = torch.rand(1, 8)
    dist.reset(noiseless_action, step=0)
    print(dist.sample())
    print(noiseless_action, dist.mean)

    dist = OrnsteinUhlenbeckNoise()
    noiseless_action = torch.rand(1, 8)
    dist.reset(noiseless_action, step=0)
    print(dist.sample())
    print(noiseless_action, dist.mean)
