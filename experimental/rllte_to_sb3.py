from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

import gymnasium as gym
import torch as th

# ===================== load the reward module ===================== #
import sys
sys.path.append("../")
from rllte.xplore.reward import RE3
# ===================== load the reward module ===================== #

class RLeXploreCallback(BaseCallback):
    """
    A custom callback for the RLeXplore toolkit.
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreCallback, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        if isinstance(self.model, OnPolicyAlgorithm):
            self.buffer = self.model.rollout_buffer
        elif isinstance(self.model, OffPolicyAlgorithm):
            self.buffer = self.model.replay_buffer
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        obs = th.as_tensor(self.buffer.observations, device=device)
        actions = th.as_tensor(self.buffer.actions, device=device)
        rewards = th.as_tensor(self.buffer.rewards, device=device)
        dones = th.as_tensor(self.buffer.episode_starts, device=device)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        intrinsic_rewards = irs.compute(samples=dict(observations=obs, 
                                                     actions=actions, 
                                                     rewards=rewards, 
                                                     terminateds=dones,
                                                     truncateds=dones, 
                                                     next_observations=obs
                                                     ))
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #

# Parallel environments
device = 'cuda'
n_envs = 4
vec_env = make_vec_env("CartPole-v1", n_envs=n_envs)

# ===================== build the reward ===================== #
irs = RE3(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space,
    n_envs=n_envs,
    device=device
)
# ===================== build the reward ===================== #

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000, callback=RLeXploreCallback(irs))