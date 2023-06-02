from collections import deque
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.base_agent import BaseAgent
from rllte.common import utils

class OffPolicyAgent(BaseAgent):
    """Trainer for off-policy algorithms.

    Args:
        env (Env): A Gym-like environment for training.
        eval_env (Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.
        num_init_steps (int): Number of initial exploration steps.
        eval_every_steps (int): Evaluation interval.

    Returns:
        Off-policy agent instance.
    """
    def __init__(self, 
                 env: gym.Env,
                 eval_env: Optional[gym.Env] = None,
                 tag: str = "default",
                 seed: int = 1,
                 device: str = "cpu",
                 pretraining: bool = False,
                 num_init_steps: int = 2000,
                 eval_every_steps: int = 5000,
                 ) -> None:
        super().__init__(env=env,
                         eval_env=eval_env,
                         tag=tag,
                         seed=seed,
                         device=device,
                         pretraining=pretraining
                         )
        self.eval_every_steps = eval_every_steps
        self.num_init_steps = num_init_steps

    def train(self, num_train_steps: int = 100000, init_model_path: Optional[str] = None) -> None:
        """Training function.
        
        Args:
            num_train_steps (int): Number of training steps.
            init_model_path (Optional[str]): Path of Iinitial model parameters.

        Returns:
            None.
        """
        # freeze the structure of the agent
        self.freeze()
        # final check
        self.check()
        # load initial model parameters 
        if init_model_path is not None:
            self.load(init_model_path)
        # reset the env
        episode_step, episode_reward = 0, 0
        obs, info = self.env.reset(seed=self.seed)
        metrics = None

        while self.global_step <= num_train_steps:
            # try to eval
            if (self.global_step % self.eval_every_steps) == 0 and (self.eval_env is not None):
                eval_metrics = self.eval()
                self.logger.eval(msg=eval_metrics)

            # sample actions
            with th.no_grad(), utils.eval_mode(self):
                action = self.act(obs, training=True)
                # Initial exploration
                if self.global_step < self.num_init_steps:
                    action.uniform_(-1.0, 1.0)
            next_obs, reward, terminated, truncated, info = self.env.step(action.clamp(*self.action_range))
            episode_reward += reward[0].cpu().numpy()
            episode_step += 1
            self.global_step += 1

            # save transition
            self.storage.add(
                obs[0].cpu().numpy(),
                action[0].cpu().numpy(),
                np.zeros_like(reward[0].cpu().numpy())
                if self.pretraining
                else reward[0].cpu().numpy(),  # pre-training mode
                terminated[0].cpu().numpy(),
                info,
                next_obs[0].cpu().numpy(),
            )

            # update agent
            if self.global_step >= self.num_init_steps:
                metrics = self.update()
                # try to update storage
                self.storage.update(metrics)

            # done
            if terminated or truncated:
                episode_time, total_time = self.timer.reset()
                if metrics is not None:
                    train_metrics = {
                        "step": self.global_step,
                        "episode": self.global_episode,
                        "episode_length": episode_step,
                        "episode_reward": episode_reward,
                        "fps": episode_step / episode_time,
                        "total_time": total_time,
                    }
                    self.logger.train(msg=train_metrics)

                obs, info = self.env.reset(seed=self.seed)
                self.global_episode += 1
                episode_step, episode_reward = 0, 0
                continue

            obs = next_obs

        # save model
        self.logger.info("Training Accomplished!")
        self.save()
        self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()

    def eval(self) -> Dict[str, float]:
        """Evaluation function."""
        step, episode, total_reward = 0, 0, 0
        obs, info = self.eval_env.reset(seed=self.seed)

        while episode <= self.num_eval_episodes:
            with th.no_grad(), utils.eval_mode(self):
                action = self.act(obs, training=False)

            next_obs, reward, terminated, truncated, info = self.eval_env.step(action.clamp(*self.action_range))
            total_reward += reward[0].cpu().numpy()
            step += 1

            if terminated or truncated:
                obs, info = self.eval_env.reset(seed=self.seed)
                episode += 1
                continue

            obs = next_obs

        return {
            "step": self.global_step,
            "episode": self.global_episode,
            "episode_length": step / episode,
            "episode_reward": total_reward / episode,
            "total_time": self.timer.total_time(),
        }