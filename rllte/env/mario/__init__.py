from typing import Callable
import gymnasium as gym
import gym as gym_old
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from rllte.env.utils import Gymnasium2Torch
from rllte.env.mario.wrappers import (
    EpisodicLifeEnv, 
    SkipFrame,
    Gym2Gymnasium,
    ImageTranspose
)

def make_mario_env(
        env_id: str = "SuperMarioBros-1-1-v3",
        num_envs: int = 8,
        device: str = "cpu",
        asynchronous: bool = True,
        seed: int = 0,
        gray_scale: bool = False,
        frame_stack: int = 0,
    ) -> Gymnasium2Torch:

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env = gym_old.make(env_id, apply_api_compatibility=True, render_mode="rgb_array")
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = Gym2Gymnasium(env)
            env = SkipFrame(env, skip=4)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            if gray_scale:
                env = gym.wrappers.GrayScaleObservation(env)
            if frame_stack > 0:
                env = gym.wrappers.FrameStack(env, frame_stack)
            if not gray_scale and frame_stack <= 0:
                env = ImageTranspose(env)
            env = EpisodicLifeEnv(env)
            env = gym.wrappers.TransformReward(env, lambda r: 0.01*r)
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
    
def make_mario_multilevel_env(
        env_id: str = "SuperMarioBrosRandomStages-v3",
        num_envs: int = 8,
        device: str = "cpu",
        asynchronous: bool = True,
        seed: int = 0,
    ) -> Gymnasium2Torch:
    
    def make_multilevel_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env = gym_old.make(
                env_id, 
                apply_api_compatibility=True,
                render_mode="rgb_array",
                stages=[
                    '1-1', '1-2', '1-4', 
                    '2-1', '2-3', '2-4',
                    '3-1', '3-2', '3-4',
                    '4-1', '4-3', '4-4',
                ]
            )
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = Gym2Gymnasium(env)
            env = SkipFrame(env, skip=4)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = ImageTranspose(env)
            env = gym.wrappers.TransformReward(env, lambda r: 0.01*r)
            env.observation_space.seed(seed)
            return env
        return _thunk
    
    envs = [make_multilevel_env(env_id, seed + i) for i in range(num_envs)]
    if asynchronous:
        envs = AsyncVectorEnv(envs)
    else:
        envs = SyncVectorEnv(envs)
    
    envs = RecordEpisodeStatistics(envs)
    return Gymnasium2Torch(envs, device=device)