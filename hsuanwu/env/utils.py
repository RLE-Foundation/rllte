import torch
import gym

from hsuanwu.common.typing import *

class TorchVecEnvWrapper(gym.Wrapper):
    """Build environments that output torch tensors.
    
    Args:
        env: The environment.
        device: Device (cpu, cuda, ...) on which the code should be run.
    
    Returns:
        Environment instance.
    """
    def __init__(self, env: Env, device: torch.device) -> None:
        super().__init__(env)
        
        self._venv = env
        self._device = torch.device(device)
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space


    def reset(self) -> Any:
        obs = self._venv.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        return obs
    

    def step(self, actions: Tensor) -> Tuple[Any]:
        self.step_async(actions)
        return self.step_wait()
    

    def step_async(self, actions: Tensor) -> None:
        if actions.dtype is torch.int64:
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self._venv.step_async(actions)
    

    def step_wait(self) -> None:
        obs, reward, done, info = self._venv.step_wait()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self._device).unsqueeze(dim=1)
        done = torch.as_tensor([[1.0] if _ else [0.0] for _ in done], dtype=torch.float32, device=self._device)

        return obs, reward, done, info