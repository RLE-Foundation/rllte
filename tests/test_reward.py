import os
import sys
import pytest

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from rllte.env import make_atari_env, make_dmc_env
from rllte.xplore.reward import (GIRM,
                                 ICM,
                                 NGU,
                                 PseudoCounts,
                                 RE3,
                                 RIDE,
                                 RISE,
                                 RND,
                                 REVD)
import torch as th

@pytest.mark.parametrize("reward", [GIRM,
                                    ICM,
                                    NGU,
                                    PseudoCounts,
                                    RE3,
                                    RIDE,
                                    RISE,
                                    RND,
                                    REVD])
@pytest.mark.parametrize("env_cls", [make_atari_env, make_dmc_env])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_reward(reward, env_cls, device):
    env = env_cls(device=device, num_envs=1)
    device = th.device(device)
    env.reset()
    irs = reward(observation_space=env.observation_space, action_space=env.action_space, device=device)
    
    obs = th.rand(size=(256, 1, *env.observation_space.shape)).to(device)
    if env_cls is make_atari_env:
        action = th.randint(0, env.action_space.n, (256, 1)).to(device)
    if env_cls is make_dmc_env:
        action = th.rand(size=(256, 1, env.action_space.shape[0])).to(device)

    samples = {
        'obs': obs,
        'actions': action,
        'next_obs': obs,
    }

    irs.compute_irs(samples)

    print("Intrinsic reward test passed!")
