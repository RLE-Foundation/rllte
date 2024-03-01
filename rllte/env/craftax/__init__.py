from typing import Callable
import gymnasium as gym
from brax.io import torch as brax_torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
from craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv

from environment_base.wrappers import (
    LogWrapper,
    BatchEnvWrapper,
    OptimisticResetVecEnvWrapper,
)

from rllte.env.craftax.wrappers import TorchWrapper, ResizeTorchWrapper, RecordEpisodeStatistics4Craftax

def make_craftax_env(
        env_id: str = "Craftax-Classic",
        num_envs: int = 8,
        device: str = "cpu",
    ):

    if env_id == "Craftax-Classic":
        env = CraftaxClassicPixelsEnv()
    elif env_id == "Craftax":
        env = CraftaxPixelsEnv()
    else:
        raise ValueError(f"Unknown environment: {env_id}")
    
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(env, num_envs=num_envs, reset_ratio=16)
    env = TorchWrapper(env, device=device)
    env = ResizeTorchWrapper(env, (84, 84))
    env = RecordEpisodeStatistics4Craftax(env)
    env.num_envs = num_envs
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    return env

