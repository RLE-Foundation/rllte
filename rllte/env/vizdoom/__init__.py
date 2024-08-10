from typing import Callable, Dict

import gymnasium as gym
import gym as gym_old
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers import NormalizeReward, TransformReward

import numpy as np
from rllte.env.utils import Gymnasium2Torch
from rllte.env.utils import EnvPoolAsync2Gymnasium, EnvPoolSync2Gymnasium, Gymnasium2Torch

def make_envpool_vizdoom_env(
    env_id: str = "MyWayHome-v1", num_envs: int = 8, device: str = "cpu", seed: int = 1, asynchronous: bool = True
) -> Gymnasium2Torch:
    """Create Atari environments with `envpool`.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device to convert the data.
        seed (int): Random seed.
        asynchronous (bool): `True` for creating asynchronous environments,
            and `False` for creating synchronous environments.

    Returns:
        The vectorized environments.
    """
    env_kwargs = dict(
        task_id=env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        batch_size=num_envs,
        seed=seed,
        episodic_life=True,
        use_combined_action=True,
        stack_num=4,
        cfg_path="/home/mila/r/roger.creus-castanyer/rllte/rllte/env/vizdoom/my_way_home.cfg",
    )

    if asynchronous:
        envs = EnvPoolAsync2Gymnasium(env_kwargs)
    else:
        envs = EnvPoolSync2Gymnasium(env_kwargs)

    #if "MyWayHome" in env_id:
    #    envs = NormalizeReward(envs, gamma=0.99)
    #    envs = TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    envs = RecordEpisodeStatistics(envs)

    return Gymnasium2Torch(envs, device, envpool=True)