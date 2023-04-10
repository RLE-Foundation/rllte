import os
import sys

import numpy as np
import torch

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import (
    make_atari_env,
    make_dmc_env,
    make_minigrid_env,
    make_procgen_env,
)

if __name__ == "__main__":
    # Atari games
    seed = 1
    envs = make_atari_env(env_id="Alien-v5", num_envs=3, seed=seed, frame_stack=4)
    print(envs.observation_space, envs.action_space)
    print(envs.action_space.sample())

    obs, info = envs.reset(seed=seed)
    print(obs.size(), info)
    for step in range(1024):
        obs, reward, terminated, truncated, info = envs.step(
            torch.randint(low=0, high=18, size=(3, 1))
        )
        if "episode" in info:
            print(info["episode"])
    envs.close()
    print("Atari games passed!")

    # Procgen games
    seed = 17
    envs = make_procgen_env(
        env_id="bigfish",
        num_envs=3,
        num_levels=200,
        start_level=0,
        distribution_mode="easy",
    )
    print(envs.observation_space, envs.action_space)
    print(envs.action_space.sample())

    obs, info = envs.reset(seed=seed)
    print(obs.size(), info)
    for step in range(1024):
        obs, reward, terminated, truncated, info = envs.step(
            torch.randint(low=0, high=15, size=(3, 1))
        )
        if "episode" in info:
            print(info["episode"])
    envs.close()
    print("Procgen games passed!")

    # DeepMind Control Suite
    seed = 7
    envs = make_dmc_env(
        env_id="cartpole_balance",
        num_envs=3,
        visualize_reward=False,
        from_pixels=True,
        frame_stack=3,
        seed=seed,
    )
    print(envs.observation_space, envs.action_space)
    print(envs.action_space.sample())

    obs, info = envs.reset(seed=seed)
    print(obs.size(), info)

    for step in range(1000):
        obs, reward, terminated, truncated, info = envs.step(
            torch.rand(size=(3, envs.action_space.shape[0]))
        )
        # print(reward, terminated, truncated)
        if "episode" in info:
            print(info['episode'])

    envs.close()
    print("DeepMind Control Suite passed!")

    # MiniGrid
    seed = 7
    envs = make_minigrid_env(
        env_id="MiniGrid-MultiRoom-N4-S5-v0",
        num_envs=1,
        fully_observable=True,
        seed=seed,
        frame_stack=1,
        device="cuda",
    )

    print(envs.observation_space, envs.action_space)
    obs, info = envs.reset(seed=seed)
    print(obs.shape, info)

    for step in range(7):
        obs, reward, terminated, truncated, info = envs.step(
            torch.randint(low=0, high=7, size=(3, 1))
        )
        print(reward, terminated)
    envs.close()
    print("MiniGrid passed!")
