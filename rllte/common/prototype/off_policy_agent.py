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
from copy import deepcopy
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch as th

from rllte.common import utils
from rllte.common.prototype.base_agent import BaseAgent
from rllte.common.type_alias import OffPolicyType, ReplayStorageType, VecEnv


class OffPolicyAgent(BaseAgent):
    """Trainer for off-policy algorithms.

    Args:
        env (VecEnv): Vectorized environments for training.
        eval_env (Optional[VecEnv]): Vectorized environments for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.
        num_init_steps (int): Number of initial exploration steps.
        **kwargs: Arbitrary arguments such as `batch_size` and `hidden_dim`.

    Returns:
        Off-policy agent instance.
    """

    def __init__(
        self,
        env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_init_steps: int = 2000,
        **kwargs,
    ) -> None:
        super().__init__(env=env, eval_env=eval_env, tag=tag, seed=seed, device=device, pretraining=pretraining)
        self.num_init_steps = num_init_steps
        # attr annotations
        self.policy: OffPolicyType
        self.storage: ReplayStorageType

    def update(self) -> None:
        """Update the agent. Implemented by individual algorithms."""
        raise NotImplementedError

    def train( # noqa: C901
        self,
        num_train_steps: int,
        init_model_path: Optional[str] = None,
        log_interval: int = 1,
        eval_interval: int = 5000,
        save_interval: int = 5000,
        num_eval_episodes: int = 10,
        th_compile: bool = False,
        anneal_lr: bool = False
    ) -> None:
        """Training function.

        Args:
            num_train_steps (int): The number of training steps.
            init_model_path (Optional[str]): The path of the initial model.
            log_interval (int): The interval of logging.
            eval_interval (int): The interval of evaluation.
            save_interval (int): The interval of saving model.
            num_eval_episodes (int): The number of evaluation episodes.
            th_compile (bool): Whether to use `th.compile` or not.
            anneal_lr (bool): Whether to anneal the learning rate or not.

        Returns:
            None.
        """
        # freeze the agent and get ready for training
        self.freeze(init_model_path=init_model_path, th_compile=th_compile)

        # reset the env
        episode_rewards: Deque = deque(maxlen=10)
        episode_steps: Deque = deque(maxlen=10)
        obs, infos = self.env.reset(seed=self.seed)

        # training loop
        while self.global_step < num_train_steps:
            # try to eval
            if (self.global_step % eval_interval) == 0 and (self.eval_env is not None):
                eval_metrics = self.eval(num_eval_episodes)

                # log to console
                self.logger.eval(msg=eval_metrics)

            # sample actions
            with th.no_grad(), utils.eval_mode(self):
                # Initial exploration
                if self.global_step < self.num_init_steps:
                    actions = th.stack([th.as_tensor(self.action_space.sample()) for _ in range(self.num_envs)])
                else:
                    actions = self.policy(obs, training=True)
            
            # update the learning rate
            if anneal_lr:
                for key in self.policy.optimizers.keys():
                    utils.linear_lr_scheduler(self.policy.optimizers[key], self.global_step, num_train_steps, self.lr)

            # update agent
            if self.global_step >= self.num_init_steps:
                self.update()
                # try to update storage
                self.storage.update(self.metrics)

            # observe reward and next obs
            next_obs, rews, terms, truncs, infos = self.env.step(actions)

            # pre-training mode
            if self.pretraining:
                rews = th.zeros_like(rews, device=self.device)

            # TODO: get real next observations
            # As the vector environments autoreset for a terminating and truncating sub-environments,
            # the returned observation and info is not the final step's observation or info which
            # is instead stored in info as `final_observation` and `final_info`. So we need to get
            # the real next observations from the infos and not to reset the environments.
            real_next_obs = deepcopy(next_obs)
            for idx, (term, trunc) in enumerate(zip(terms, truncs)):
                if term.item() or trunc.item():
                    # TODO: deal with dict observations
                    real_next_obs[idx] = th.as_tensor(infos["final_observation"][idx], device=self.device) # type: ignore[index]

            # add new transitions
            self.storage.add(obs, actions, rews, terms, truncs, infos, real_next_obs)
            self.global_step += self.num_envs

            # deal with the intrinsic reward module
            # for modules like RE3, this will calculate the random embeddings
            # and insert them into the storage. for modules like ICM, this
            # will update the dynamic models.
            if self.irs is not None:
                self.irs.add(samples={"obs": obs, "actions": actions, "next_obs": real_next_obs})  # type: ignore

            # get episode information
            eps_r, eps_l = utils.get_episode_statistics(infos)
            episode_rewards.extend(eps_r)
            episode_steps.extend(eps_l)
            self.global_episode += len(eps_r)

            # log training information
            if len(episode_rewards) >= 1 and (self.global_step % log_interval) == 0:
                total_time = self.timer.total_time()

                # log to console
                train_metrics = {
                    "step": self.global_step,
                    "episode": self.global_episode,
                    "episode_length": np.mean(list(episode_steps)),
                    "episode_reward": np.mean(list(episode_rewards)),
                    "fps": self.global_step / total_time,
                    "total_time": total_time,
                }
                self.logger.train(msg=train_metrics)

            # set the current observation
            obs = next_obs

            # save model
            if self.global_step % save_interval == 0:
                self.save()

        # final save
        self.save()
        self.logger.info("Training Accomplished!")
        self.logger.info(f"Model saved at: {self.work_dir / 'model'}")

        # close env
        self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()

    def eval(self, num_eval_episodes: int) -> Dict[str, Any]:
        """Evaluation function.

        Args:
            num_eval_episodes (int): The number of evaluation episodes.

        Returns:
            The evaluation results.
        """
        assert self.eval_env is not None, "No evaluation environment is provided!"
        # reset the env
        obs, infos = self.eval_env.reset(seed=self.seed)
        episode_rewards: List[float] = []
        episode_steps: List[int] = []

        # evaluation loop
        while len(episode_rewards) < num_eval_episodes:
            # sample actions
            with th.no_grad(), utils.eval_mode(self):
                actions = self.policy(obs, training=False)

            # observe reward and next obs
            next_obs, rews, terms, truncs, infos = self.eval_env.step(actions)

            # get episode information
            if "episode" in infos:
                eps_r, eps_l = utils.get_episode_statistics(infos)
                episode_rewards.extend(eps_r)
                episode_steps.extend(eps_l)

            # set the current observation
            obs = next_obs

        return {
            "step": self.global_step,
            "episode": self.global_episode,
            "episode_length": np.mean(episode_steps),
            "episode_reward": np.mean(episode_rewards),
            "total_time": self.timer.total_time(),
        }
