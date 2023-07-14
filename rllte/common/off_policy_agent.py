# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from pathlib import Path
from typing import Dict, Optional
from collections import deque

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common import utils
from rllte.common.base_agent import BaseAgent


class OffPolicyAgent(BaseAgent):
    """Trainer for off-policy algorithms.

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.
        num_init_steps (int): Number of initial exploration steps.
        eval_every_steps (int): Evaluation interval.
        **kwargs: Arbitrary arguments such as `batch_size` and `hidden_dim`.

    Returns:
        Off-policy agent instance.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_init_steps: int = 2000,
        eval_every_steps: int = 5000,
    ) -> None:
        super().__init__(env=env, eval_env=eval_env, tag=tag, seed=seed, device=device, pretraining=pretraining)

        self.eval_every_steps = eval_every_steps
        self.num_init_steps = num_init_steps

    def update(self) -> Dict[str, float]:
        """Update the agent. Implemented by individual algorithms."""
        raise NotImplementedError

    def freeze(self) -> None:
        """Freeze the structure of the agent. Implemented by individual algorithms."""
        # freeze the policy
        self.policy.freeze(encoder=self.encoder, dist=self.dist)
        # initialize the policy
        self.policy.apply(self.network_init_method)
        # to device
        self.policy.to(self.device)
        # set the training mode
        self.mode(training=True)

    def mode(self, training: bool = True) -> None:
        """Set the training mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self.policy.train(training)

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
            self.logger.info(f"Loading Initial Parameters from {init_model_path}...")
            self.policy.load(init_model_path, self.device)

        # reset the env
        episode_rewards = deque(maxlen=10)
        episode_steps = deque(maxlen=10)
        train_metrics = {}
        time_step = self.env.reset(seed=self.seed)

        # training loop
        while self.global_step <= num_train_steps:
            # try to eval
            if (self.global_step % self.eval_every_steps) == 0 and (self.eval_env is not None):
                eval_metrics = self.eval()
                # write to tensorboard
                self.writer.add_scalar("Evaluation/Average Episode Reward", eval_metrics["episode_reward"], self.global_step)
                self.writer.add_scalar("Evaluation/Average Episode Length", eval_metrics["episode_length"], self.global_step)

            # sample actions
            with th.no_grad(), utils.eval_mode(self):
                # Initial exploration
                if self.global_step <= self.num_init_steps:
                    actions = th.rand(size=(self.num_envs, self.action_dim), device=self.device).uniform_(-1.0, 1.0)
                else:
                    actions = self.policy(time_step.observation, training=True, step=self.global_step)

            # observe reward and next obs
            time_step = self.env.step(actions)
            self.global_step += self.num_envs

            # pre-training mode
            if self.pretraining:
                time_step = time_step._replace(reward=th.zeros_like(time_step.reward, device=self.device))

            # add new transitions
            self.storage.add(*time_step)

            # update agent
            if self.global_step >= self.num_init_steps:
                train_metrics = self.update()
                # try to update storage
                self.storage.update(train_metrics)

            # get episode information
            if "episode" in time_step.info:
                eps_r, eps_l = time_step.get_episode_statistics()
                episode_rewards.extend(eps_r)
                episode_steps.extend(eps_l)
                self.global_episode += len(eps_r)

            # log training information
            if len(episode_rewards) > 1:
                # write to tensorboard
                total_time = self.timer.total_time()
                for key in train_metrics.keys():
                    self.writer.add_scalar(f"Training/{key}", train_metrics[key], self.global_step)
                self.writer.add_scalar('Training/Average Episode Reward', np.mean(episode_rewards), self.global_step)
                self.writer.add_scalar('Training/Average Episode Length', np.mean(episode_steps), self.global_step)
                self.writer.add_scalar("Training/Number of Episodes", self.global_episode, self.global_step)
                self.writer.add_scalar('Training/FPS', self.global_step / total_time, self.global_step)
                self.writer.add_scalar('Training/Total Time', total_time, self.global_step)

                # As the vector environments autoreset for a terminating and truncating sub-environments,
                # the returned observation and info is not the final step's observation or info which
                # is instead stored in info as `final_observation` and `final_info`. Therefore,
                # we don't need to reset the env here.

            # set the current observation
            time_step = time_step._replace(observation=time_step.next_observation)

        # save model
        self.logger.info("Training Accomplished!")
        if self.pretraining:  # pretraining
            save_dir = Path.cwd() / "pretrained"
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = Path.cwd() / "model"
            save_dir.mkdir(exist_ok=True)
        self.policy.save(path=save_dir, pretraining=self.pretraining)
        self.logger.info(f"Model saved at: {save_dir}")

        # close env
        self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()

    def eval(self) -> Dict[str, float]:
        """Evaluation function."""
        # reset the env
        time_step = self.eval_env.reset(seed=self.seed)
        episode_rewards = list()
        episode_steps = list()

        # evaluation loop
        while len(episode_rewards) < self.num_eval_episodes:
            # sample actions
            with th.no_grad(), utils.eval_mode(self):
                actions = self.policy(time_step.observation, training=False, step=self.global_step)

            # observe reward and next obs
            time_step = self.eval_env.step(actions)

            # get episode information
            if "episode" in time_step.info:
                eps_r, eps_l = time_step.get_episode_statistics()
                episode_rewards.extend(eps_r)
                episode_steps.extend(eps_l)

            # set the current observation
            time_step = time_step._replace(observation=time_step.next_observation)

        return {
            "episode_length": np.mean(episode_steps),
            "episode_reward": np.mean(episode_rewards),
        }
