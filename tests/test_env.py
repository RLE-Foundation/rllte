import os
import sys

import numpy as np
import torch

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_atari_env, make_dmc_env, make_procgen_env

if __name__ == "__main__":
    # Atari games
    envs = make_atari_env(env_id="Alien-v5", num_envs=3, seed=0, frame_stack=4)
    print(envs.observation_space, envs.action_space)
    obs, info = envs.reset()
    print(obs.size(), info)

    for step in range(7):
        obs, reward, terminated, truncated, info = envs.step(
            torch.randint(low=0, high=18, size=(3, 1)))
        print(reward, terminated, truncated)
    print('Atari games passed!')

    # Procgen games
    envs = make_procgen_env(env_id='bigfish', 
                            num_envs=3, 
                            num_levels=200, 
                            start_level=0, 
                            distribution_mode='easy')
    print(envs.observation_space, envs.action_space)
    obs, info = envs.reset()
    print(obs.size(), info)

    for step in range(7):
        obs, reward, terminated, truncated, info = envs.step(
            torch.randint(low=0, high=15, size=(3, 1)))
        print(reward, terminated, truncated)
    print('Procgen games passed!')