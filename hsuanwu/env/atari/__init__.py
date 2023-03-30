import gymnasium as gym
import numpy as np

from gymnasium.spaces import Box
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import (
    FrameStack,
    GrayScaleObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
    TransformReward,
)

from hsuanwu.common.typing import *
from hsuanwu.env.atari.wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

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
            shape=[3, 84, 84],
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


def make_atari_env(
    env_id: str = "Alien-v5",
    num_envs: int = 8,
    seed: int = 0,
    frame_stack: int = 4,
    device: torch.device = "cuda",
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

    env_id = "ALE/" + env_id
    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    envs = SyncVectorEnv(envs)

    return TorchVecEnvWrapper(envs, device, lambda obs: obs)
