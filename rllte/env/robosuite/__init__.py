from typing import Callable, Dict, Tuple

import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from robosuite.wrappers import GymWrapper

from rllte.env.utils import TorchVecEnvWrapper


class AdapterEnv(gym.Wrapper):
    """PyBullet robotics envs currently doesn't support Gymnasium.

    Args:
        env (Env): Environment to wrap.

    Returns:
        AdapterEnv instance.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
            low=env.observation_space.low,
            high=env.observation_space.high,
        )
        self.action_space = gym.spaces.Box(
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
            low=env.action_space.low,
            high=env.action_space.high,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, done, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs = self.env.reset()
        return obs, {}


def make_robosuite_env(
    env_id: str = "Lift_Panda",
    num_envs: int = 1,
    device: str = "cpu",
    seed: int = 0,
    distributed: bool = False,
    has_renderer: bool = False,
    has_offscreen_renderer: bool = False,
    use_camera_obs: bool = False,
) -> gym.Env:
    """Build Robosuite robotics environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        seed (int): Random seed.
        distributed (bool): For `Distributed` algorithms, in which `SyncVectorEnv` is required
            and reward clip will be used before environment vectorization.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering.
        use_camera_obs (bool): True for using image observations.

    Returns:
        Environments instance.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env_name, robots = env_id.split("_")
            env = suite.make(
                env_name=env_name,
                robots=robots,
                has_renderer=has_renderer,
                has_offscreen_renderer=has_offscreen_renderer,
                use_camera_obs=use_camera_obs,
            )
            env = GymWrapper(env)
            env.seed(seed)
            env = AdapterEnv(env)
            env.observation_space.seed(seed)
            env.action_space.seed(seed)
            return env

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    if distributed:
        envs = SyncVectorEnv(envs)
    else:
        envs = AsyncVectorEnv(envs)
        envs = RecordEpisodeStatistics(envs)

    return TorchVecEnvWrapper(envs, device=device)
