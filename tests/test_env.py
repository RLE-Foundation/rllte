import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_procgen_env, make_atari_env, make_dmc_env
import numpy as np

if __name__ == '__main__':
    # envs = make_procgen_env(env_id='bigfish', num_envs=2)
    # print(envs.observation_space, envs.action_space)
    # obs = envs.reset()

    # for step in range(256):
    #     obs, reward, done, info = envs.step(np.random.randint(0, 15, size=(2, )))
    #     print(reward, done)
    
    envs = make_atari_env(env_id='Alien-v5', num_envs=8, seed=0, frame_stack=4)
    print(envs.observation_space, envs.single_action_space)
    obs = envs.reset()

    for step in range(256):
        obs, reward, done, info = envs.step(np.random.randint(0, 18, size=(8, )))
        print(reward, done)