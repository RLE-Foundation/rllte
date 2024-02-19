from typing import Callable, Dict
import gym as gym_old
import griddly
import gymnasium as gym

from griddly import GymWrapperFactory, gd
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from rllte.env.utils import Gymnasium2Torch
from gymnasium.wrappers import RecordEpisodeStatistics
from rllte.env.griddly.wrappers import Gym2Gymnasium, ImageTranspose, GymnasiumGriddlyEnv, register_griddly_envs


def make_griddly_env(
        env_id: str = "MazeEnv",
        num_envs: int = 8,
        device: str = "cpu",
        asynchronous: bool = False,
        seed: int = 0,
    ) -> Gymnasium2Torch:

    register_griddly_envs()

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env = gym_old.make(
                env_id,
                apply_api_compatibility=True,
                player_observer_type=gd.ObserverType.VECTOR,
                global_observer_type=gd.ObserverType.SPRITE_2D,
                render_mode="rgb_array"
            )
            env.reset()
            obs_shape = env.unwrapped.env.render(observer="global", mode="rgb_array").shape
            env = GymnasiumGriddlyEnv(env, obs_shape)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = ImageTranspose(env)
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