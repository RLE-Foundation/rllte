from gym.wrappers import (RecordEpisodeStatistics,
                          ResizeObservation,
                          GrayScaleObservation,
                          FrameStack,
                          TransformReward)
from gym.vector import SyncVectorEnv

import gym
import numpy as np

from hsuanwu.common.typing import *
from hsuanwu.env.atari.wrappers import (NoopResetEnv,
                                        MaxAndSkipEnv,
                                        EpisodicLifeEnv,
                                        FireResetEnv)
from hsuanwu.env.utils import TorchVecEnvWrapper

def make_atari_env(env_id: str = 'Alien-v5',
                   num_envs: int = 8,
                   seed: int = 0,
                   frame_stack: int = 4,
                   device: torch.device = 'cuda'
                   ) -> Env:
    """Build Atari environments.

    Args:
        env_id: Name of environment.
        num_envs: Number of parallel environments.
        seed: Random seed.
        frame_stack: Number of stacked frames.
        device: Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        Environments instance.
    """
    def make_env(env_id: str, seed: int) -> Env:
        def _thunk():
            env = gym.make(env_id)
            env = RecordEpisodeStatistics(env)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=frame_stack)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)

            env = TransformReward(env, lambda reward: np.sign(reward))
            env = ResizeObservation(env, shape=(84, 84))
            env = GrayScaleObservation(env)
            env = FrameStack(env, frame_stack)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            return env

        return _thunk

    env_id = 'ALE/' + env_id
    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    envs = SyncVectorEnv(envs)

    return TorchVecEnvWrapper(envs, device, lambda obs: obs)