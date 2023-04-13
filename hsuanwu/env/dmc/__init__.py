import gymnasium as gym
import torch
from gymnasium.envs.registration import register
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from hsuanwu.common.typing import Callable, Device, Dict, Env, List, Tensor, Tuple
from hsuanwu.env.utils import FrameStack


class TorchVecEnvWrapper(gym.Wrapper):
    """Build environments that output torch tensors.

    Args:
        env (Env): The environment.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        TorchVecEnv instance.
    """

    def __init__(self, env: Env, device: Device) -> None:
        super().__init__(env)
        self._device = torch.device(device)
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.num_envs = len(env.envs)

    def reset(self, **kwargs) -> Tuple[Tensor, Dict]:
        obs, info = self.env.reset(**kwargs)
        obs = torch.as_tensor(obs, device=self._device)
        return obs, info

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
        obs = torch.as_tensor(obs, device=self._device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self._device)
        terminated = torch.as_tensor(
            [[1.0] if _ else [0.0] for _ in terminated],
            dtype=torch.float32,
            device=self._device,
        )
        truncated = torch.as_tensor(
            [[1.0] if _ else [0.0] for _ in truncated],
            dtype=torch.float32,
            device=self._device,
        )

        return obs, reward, terminated, truncated, info


def make_dmc_env(
    env_id: str = "cartpole_balance",
    num_envs: int = 1,
    device: Device = "cuda",
    resource_files: List = None,
    img_source: str = None,
    total_frames: int = None,
    seed: int = 1,
    visualize_reward: bool = False,
    from_pixels: bool = True,
    height: int = 84,
    width: int = 84,
    camera_id: int = 0,
    frame_stack: int = 3,
    frame_skip: int = 2,
    episode_length: int = 1000,
    environment_kwargs: Dict = None,
):
    """Build DeepMind Control Suite environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of parallel environments.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        resource_files (List): File path of the resource files.
        img_source (str): Type of the background distractor, supported values: ['color', 'noise', 'images', 'video'].
        total_frames (int): for 'images' or 'video' distractor.
        seed (int): Random seed.
        visualize_reward (bool): True when 'from_pixels' is False, False when 'from_pixels' is True.
        from_pixels (bool): Provide image-based observations or not.
        height (int): Image observation height.
        width (int): Image observation width.
        camera_id (int): Camera id for generating image-based observations.
        frame_stack (int): Number of stacked frames.
        frame_skip (int): Number of action repeat.
        episode_length (int): Maximum length of an episode.
        environment_kwargs (Dict): Other environment arguments.

    Returns:
        Environments instance.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            domain_name, task_name = env_id.split("_")
            _env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seed)

            if from_pixels:
                assert (
                    not visualize_reward
                ), "Cannot use visualize reward when learning from pixels!"

            # shorten episode length
            max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

            if not _env_id in gym.envs.registry:
                register(
                    id=env_id,
                    entry_point="hsuanwu.env.dmc.wrappers:DMCWrapper",
                    kwargs={
                        "domain_name": domain_name,
                        "task_name": task_name,
                        "resource_files": resource_files,
                        "img_source": img_source,
                        "total_frames": total_frames,
                        "task_kwargs": {"random": seed},
                        "environment_kwargs": environment_kwargs,
                        "visualize_reward": visualize_reward,
                        "from_pixels": from_pixels,
                        "height": height,
                        "width": width,
                        "camera_id": camera_id,
                        "frame_skip": frame_skip,
                    },
                    max_episode_steps=max_episode_steps,
                )

            if visualize_reward:
                return gym.make(env_id)
            else:
                return FrameStack(gym.make(env_id), frame_stack)

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return TorchVecEnvWrapper(envs, device)
