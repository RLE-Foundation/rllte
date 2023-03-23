from gym.wrappers import (TransformObservation,
                          RecordEpisodeStatistics,
                          NormalizeReward,
                          TransformReward
)
from procgen import ProcgenEnv

import numpy as np

from hsuanwu.common.typing import *
from hsuanwu.env.utils import TorchVecEnvWrapper





def make_procgen_env(env_id: str = 'bigfish',
                     num_envs: int = 64,
                     gamma: float = 0.99,
                     num_levels: int = 0,
                     start_level: int = 0,
                     distribution_mode: str = "easy",
                     device: torch.device = 'cuda'
                     ) -> Env:
    """Build Prcogen environments.

    Args:
        env_id: Name of environment.
        num_envs: Number of parallel environments.
        gamma: A discount factor.
        num_levels: The number of unique levels that can be generated. Set to 0 to use unlimited levels.
        start_level: The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
        distribution_mode: What variant of the levels to use, the options are "easy", "hard", "extreme", "memory", "exploration".
        device: Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        Environments instance.
    """
    envs = ProcgenEnv(num_envs=num_envs, 
                      env_name=env_id, 
                      num_levels=num_levels, 
                      start_level=start_level, 
                      distribution_mode=distribution_mode)
    envs = TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = RecordEpisodeStatistics(envs)
    envs = NormalizeReward(envs, gamma=gamma)
    envs = TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    return TorchVecEnvWrapper(envs, device)