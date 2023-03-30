import numpy as np
from gym.spaces.box import Box
from gym.wrappers import (
    NormalizeReward,
    RecordEpisodeStatistics,
    TransformObservation,
    TransformReward,
)
from procgen import ProcgenEnv

from hsuanwu.common.typing import *


class TorchVecEnvWrapper:
    """Build environments that output torch tensors.

    Args:
        env: The environment.
        device: Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        Environment instance.
    """

    def __init__(self, env: Env, device: torch.device) -> None:
        self._venv = env
        self._device = torch.device(device)
        self.observation_space = Box(
            low=env.single_observation_space.low[0, 0, 0],
            high=env.single_observation_space.high[0, 0, 0],
            shape=[3, 64, 64],
            dtype=env.single_observation_space.dtype,
        )
        self.action_space = env.single_action_space

    def reset(self) -> Any:
        obs = self._venv.reset()
        obs = torch.as_tensor(
            obs.transpose(0, 3, 1, 2), dtype=torch.float32, device=self._device
        )
        return obs

    def step(self, actions: Tensor) -> Tuple[Any]:
        if actions.dtype is torch.int64:
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()

        obs, reward, done, info = self._venv.step(actions)
        obs = torch.as_tensor(
            obs.transpose(0, 3, 1, 2), dtype=torch.float32, device=self._device
        )
        reward = torch.as_tensor(
            reward, dtype=torch.float32, device=self._device
        ).unsqueeze(dim=1)
        done = torch.as_tensor(
            [[1.0] if _ else [0.0] for _ in done],
            dtype=torch.float32,
            device=self._device,
        )

        return obs, reward, done, info


def make_procgen_env(
    env_id: str = "bigfish",
    num_envs: int = 64,
    gamma: float = 0.99,
    num_levels: int = 0,
    start_level: int = 0,
    distribution_mode: str = "easy",
    device: torch.device = "cuda",
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
    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    envs = TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = RecordEpisodeStatistics(envs)
    envs = NormalizeReward(envs, gamma=gamma)
    envs = TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    return TorchVecEnvWrapper(envs, device)
