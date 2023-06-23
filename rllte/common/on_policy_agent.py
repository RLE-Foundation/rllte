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

from collections import deque
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common import utils
from rllte.common.base_agent import BaseAgent


class OnPolicyAgent(BaseAgent):
    """Trainer for on-policy algorithms.

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.
        num_steps (int): The sample length of per rollout.
        eval_every_episodes (int): Evaluation interval.

    Returns:
        On-policy agent instance.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_steps: int = 128,
        eval_every_episodes: int = 10,
    ) -> None:
        super().__init__(env=env, eval_env=eval_env, tag=tag, seed=seed, device=device, pretraining=pretraining)
        self.num_steps = num_steps
        self.eval_every_episodes = eval_every_episodes

    def update(self) -> Dict[str, float]:
        """Update the agent. Implemented by individual algorithms."""
        raise NotImplementedError

    def freeze(self) -> None:
        """Freeze the structure of the agent. Implemented by individual algorithms."""
        # freeze the policy
        self.policy.freeze(encoder=self.encoder, dist=self.dist)
        # to device
        self.policy.to(self.device)
        # set the training mode
        self.mode(training=True)

    def mode(self, training: bool = True) -> None:
        """Set the training mode.

        Args:
            training (bool): True (training) or False (evaluation).

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
        obs, info = self.env.reset(seed=self.seed)
        # Number of updates
        num_updates = num_train_steps // self.num_envs // self.num_steps

        for update in range(num_updates):
            # try to eval
            if (update % self.eval_every_episodes) == 0 and (self.eval_env is not None):
                eval_metrics = self.eval()
                self.logger.eval(msg=eval_metrics)

            for _step in range(self.num_steps):
                # sample actions
                with th.no_grad(), utils.eval_mode(self):
                    actions, values, log_probs = self.policy.act(obs, training=True)

                (
                    next_obs,
                    rewards,
                    terminateds,
                    truncateds,
                    infos,
                ) = self.env.step(actions.clamp(*self.action_range))

                if "episode" in infos:
                    indices = np.nonzero(infos["episode"]["l"])
                    episode_rewards.extend(infos["episode"]["r"][indices].tolist())
                    episode_steps.extend(infos["episode"]["l"][indices].tolist())

                # add transitions
                self.storage.add(
                    obs=obs,
                    actions=actions,
                    rewards=th.zeros_like(rewards, device=self.device) if self.pretraining else rewards,  # pre-training mode
                    terminateds=terminateds,
                    truncateds=truncateds,
                    next_obs=next_obs,
                    log_probs=log_probs,
                    values=values,
                )

                obs = next_obs

            # get the value estimation of the last step
            with th.no_grad():
                last_values = self.policy.get_value(next_obs).detach()

            # compute intrinsic rewards
            if self.irs is not None:
                intrinsic_rewards = self.irs.compute_irs(
                    samples={
                        "obs": self.storage.obs[:-1],
                        "actions": self.storage.actions,
                        "next_obs": self.storage.obs[1:],
                    },
                    step=self.global_episode * self.num_envs * self.num_steps,
                )
                self.storage.rewards += intrinsic_rewards.to(self.device)

            # perform return and advantage estimation
            self.storage.compute_returns_and_advantages(last_values)

            # agent update
            self.update()

            # update and reset buffer
            self.storage.update()

            self.global_episode += 1
            self.global_step += self.num_envs * self.num_steps
            episode_time, total_time = self.timer.reset()

            if len(episode_rewards) > 1:
                train_metrics = {
                    "step": self.global_step,
                    "episode": self.global_episode * self.num_envs,
                    "episode_length": np.mean(episode_steps),
                    "episode_reward": np.mean(episode_rewards),
                    "fps": self.num_steps * self.num_envs / episode_time,
                    "total_time": total_time,
                }
                self.logger.train(msg=train_metrics)

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
        obs, info = self.eval_env.reset(seed=self.seed)
        episode_rewards = list()
        episode_steps = list()

        while len(episode_rewards) < self.num_eval_episodes:
            with th.no_grad(), utils.eval_mode(self):
                actions = self.policy.act(obs, training=False)
            obs, rewards, terminateds, truncateds, infos = self.eval_env.step(actions.clamp(*self.action_range))

            if "episode" in infos:
                indices = np.nonzero(infos["episode"]["l"])
                episode_rewards.extend(infos["episode"]["r"][indices].tolist())
                episode_steps.extend(infos["episode"]["l"][indices].tolist())

        return {
            "step": self.global_step,
            "episode": self.global_episode,
            "episode_length": np.mean(episode_steps),
            "episode_reward": np.mean(episode_rewards),
            "total_time": self.timer.total_time(),
        }
