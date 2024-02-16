from typing import Callable, Dict

import miniworld
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from rllte.env.utils import Gymnasium2Torch
from rllte.env.miniworld.wrappers import ImageTranspose

def make_miniworld_env(
        env_id: str = "MiniWorld-Maze-v0",
        num_envs: int = 8,
        device: str = "cpu",
        asynchronous: bool = False,
        seed: int = 0,
        num_rows: int = 32,
        num_cols: int = 32,
        room_size: int = 3,
    ) -> Gymnasium2Torch:

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                num_rows=num_rows,
                num_cols=num_cols,
                room_size=room_size,
                domain_rand=False,
                max_episode_steps=1000,
            )
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = ImageTranspose(env)
            env = gym.wrappers.TransformReward(env, lambda r: r * 10)
            env.observation_space.seed(seed)
            return env
        return _thunk
    
    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    
    if asynchronous:
        envs = AsyncVectorEnv(envs)
    else:
        envs = SyncVectorEnv(envs)
    
    envs = RecordEpisodeStatistics(envs)
    return Gymnasium2Torch(envs, device=device)