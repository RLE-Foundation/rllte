from typing import Callable, Dict

import miniworld
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from rllte.env.utils import Gymnasium2Torch
from rllte.env.miniworld.wrappers import ImageTranspose

from miniworld.params import DEFAULT_PARAMS

def make_miniworld_env(
        env_id: str = "MiniWorld-Maze-v0",
        num_envs: int = 8,
        device: str = "cpu",
        asynchronous: bool = False,
        seed: int = 0,
        num_rows: int = 5,
        num_cols: int = 5,
    ) -> Gymnasium2Torch:

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():

            # Parameters for larger movement steps, fast stepping
            default_params = DEFAULT_PARAMS.no_random()
            default_params.set("forward_step", 0.7)
            default_params.set("turn_step", 45)

            env = gym.make(
                env_id,
                render_mode="rgb_array",
                num_rows=num_rows,
                num_cols=num_cols,
                room_size=3.0,
                domain_rand=True,
                max_episode_steps=200,
                params=default_params,
            )
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = ImageTranspose(env)
            #env = gym.wrappers.TransformReward(env, lambda r: r * 10)
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