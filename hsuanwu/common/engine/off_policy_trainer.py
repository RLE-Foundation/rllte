import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import open_dict

from hsuanwu.common.engine import BasePolicyTrainer, utils
from hsuanwu.common.logger import *
from hsuanwu.common.typing import *
from hsuanwu.xploit.storage.utils import worker_init_fn


class OffPolicyTrainer(BasePolicyTrainer):
    """Trainer for off-policy algorithms.

    Args:
        train_env: A Gym-like environment for training.
        test_env: A Gym-like environment for testing.
        cfgs: Dict config for configuring RL algorithms.

    Returns:
        Off-policy trainer instance.
    """

    def __init__(self, train_env: Env, test_env: Env, cfgs: DictConfig) -> None:
        super().__init__(train_env, test_env, cfgs)
        # xploit part
        self._learner = hydra.utils.instantiate(self._cfgs.learner)
        encoder = hydra.utils.instantiate(self._cfgs.encoder).to(self._device)
        self._learner.set_encoder(encoder)
        self._replay_storage = hydra.utils.instantiate(self._cfgs.storage)

        # xplore part
        dist = hydra.utils.get_class(self._cfgs.distribution._target_)
        self._learner.set_dist(dist)
        if self._cfgs.use_aug:
            aug = hydra.utils.instantiate(self._cfgs.augmentation).to(self._device)
            self._learner.set_aug(aug)
        if self._cfgs.use_irs:
            irs = hydra.utils.instantiate(self._cfgs.reward)
            self._learner.set_irs(irs)

        # make data loader
        if self._cfgs.storage._target_ == "NStepReplayBuffer":
            self._replay_loader = torch.utils.data.DataLoader(
                self._replay_storage,
                batch_size=self._replay_storage.get_batch_size,
                num_workers=self._replay_storage.get_num_workers,
                pin_memory=self._replay_storage.get_pin_memory,
                worker_init_fn=worker_init_fn,
            )
            self._replay_iter = None

        # training track
        self._num_train_steps = self._cfgs.num_train_steps
        self._num_init_steps = self._cfgs.num_init_steps
        self._test_every_steps = self._cfgs.test_every_steps

        # debug
        self._logger.log(DEBUG, "Check Accomplished. Start Training...")

    @property
    def replay_iter(self) -> Iterable:
        """Create iterable dataloader."""
        if self._replay_iter is None:
            self._replay_iter = iter(self._replay_loader)
        return self._replay_iter

    def train(self) -> None:
        """Training function."""
        episode_step, episode_reward = 0, 0
        obs = self._train_env.reset()
        metrics = None

        while self._global_step <= self._num_train_steps:
            # try to test
            if self._global_step % self._test_every_steps == 0:
                test_metrics = self.test()
                self._logger.log(level=TEST, msg=test_metrics)

            # sample actions
            with torch.no_grad(), utils.eval_mode(self._learner):
                action = self._learner.act(obs, training=True, step=self._global_step)
            next_obs, reward, done, info = self._train_env.step(action)
            episode_reward += reward
            episode_step += 1
            self._global_step += 1

            # save transition
            self._replay_storage.add(obs, action, reward, done, info, next_obs)

            # update agent
            if self._global_step >= self._num_init_steps:
                try:
                    # TODO: for NStepReplayBuffer
                    metrics = self._learner.update(
                        self.replay_iter, step=self._global_step
                    )
                except:
                    metrics = self._learner.update(
                        self._replay_storage, step=self._global_step
                    )

            # done
            if done:
                episode_time, total_time = self._timer.reset()
                if metrics is not None:
                    train_metrics = {
                        "step": self._global_step,
                        "episode": self._global_episode,
                        "episode_length": episode_step,
                        "episode_reward": episode_reward,
                        "fps": episode_step / episode_time,
                        "total_time": total_time,
                    }
                    self._logger.log(level=TRAIN, msg=train_metrics)

                obs = self._train_env.reset()
                self._global_episode += 1
                episode_step, episode_reward = 0, 0
                continue

            obs = next_obs

    def test(self) -> None:
        """Testing function."""
        step, episode, total_reward = 0, 0, 0
        obs = self._test_env.reset()

        while episode <= self._cfgs.num_test_episodes:
            with torch.no_grad(), utils.eval_mode(self._learner):
                action = self._learner.act(obs, training=False, step=self._global_step)

            next_obs, reward, done, info = self._test_env.step(action)
            total_reward += reward
            step += 1

            if done:
                obs = self._test_env.reset()
                episode += 1
                continue

            obs = next_obs

        return {
            "step": self._global_step,
            "episode": self._global_episode,
            "episode_length": step / episode,
            "episode_reward": total_reward / episode,
            "total_time": self._timer.total_time(),
        }
