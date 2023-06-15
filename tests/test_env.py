import os
import sys
import pytest

import torch as th

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from rllte.env import (make_atari_env, 
                       make_dmc_env, 
                       make_minigrid_env,
                       make_procgen_env,
                       make_bullet_env,
                       make_robosuite_env
                       )

@pytest.mark.parametrize("env_cls", [make_atari_env, 
                                     make_minigrid_env,
                                     make_procgen_env,
                                     make_dmc_env, 
                                     make_bullet_env,
                                     make_robosuite_env])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_discrete_env(env_cls, device):
    if env_cls in [make_procgen_env, make_dmc_env]:
        env = env_cls(device=device, num_envs=1)
    else:
        env = env_cls(device=device, num_envs=1, distributed=True)
    obs, info = env.reset()

    print(env.observation_space, env.action_space)

    for step in range(10):
        action = env.action_space.sample()
        
        if env_cls in [make_atari_env, make_minigrid_env, make_procgen_env]:
            action = th.randint(0, env.action_space.n, (1,)).to(device)
        else:
            action = th.rand(size=(1, env.action_space.shape[0])).to(device)

        obs, reward, terminated, truncated, info = env.step(action)
        if "episode" in info:
            print(info["episode"])
    env.close()

    print("Environment test passed!")
    