import gymnasium as gym
import numpy as np
import torch

from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import (
    FrameStack,
    GrayScaleObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
    TransformReward,
)

from hsuanwu.common.typing import Env, Device, Any, Tensor, Tuple, Ndarray, Dict
from hsuanwu.env.atari.wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

class TorchVecEnvWrapper:
    """Build environments that output torch tensors.

    Args:
        env (Env): The environment.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        TorchVecEnv instance.
    """

    def __init__(self, env: Env, device: Device) -> None:
        self._venv = env
        self._device = torch.device(device)
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space

    def reset(self) -> Tuple[Ndarray, Dict]:
        obs, info = self._venv.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        return obs, info

    def step(self, actions: Tensor) -> Tuple[Ndarray, float, bool, bool, Dict]:
        if actions.dtype is torch.int64:
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()

        obs, reward, terminated, truncated, info = self._venv.step(actions)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        reward = torch.as_tensor(
            reward, dtype=torch.float32, device=self._device
        ).unsqueeze(dim=1)
        terminated = torch.as_tensor(
            [[1.0] if _ else [0.0] for _ in terminated],
            dtype=torch.float32,
            device=self._device,
        )

        return obs, reward, terminated, truncated, info


def make_atari_env(
    env_id: str = "Alien-v5",
    num_envs: int = 8,
    seed: int = 0,
    frame_stack: int = 4,
    device: torch.device = "cuda",
) -> Env:
    """Build Atari environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of parallel environments.
        seed (int): Random seed.
        frame_stack (int): Number of stacked frames.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.

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

    env_id = "ALE/" + env_id
    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    envs = SyncVectorEnv(envs)

    return TorchVecEnvWrapper(envs, device)
