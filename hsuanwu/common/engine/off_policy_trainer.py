from pathlib import Path
from typing import Dict

import gymnasium as gym
import hydra
import numpy as np
import omegaconf
import torch as th

from hsuanwu.common.engine.base_policy_trainer import BasePolicyTrainer
from hsuanwu.common.engine.utils import eval_mode


class OffPolicyTrainer(BasePolicyTrainer):
    """Trainer for off-policy algorithms.

    Args:
        cfgs (DictConfig): Dict config for configuring RL algorithms.
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.

    Returns:
        Off-policy trainer instance.
    """

    def __init__(self, cfgs: omegaconf.DictConfig, train_env: gym.Env, test_env: gym.Env = None) -> None:
        super().__init__(cfgs, train_env, test_env)
        self._logger.info("Deploying OffPolicyTrainer...")
        # TODO: turn on the pretraining mode, no extrinsic rewards will be provided.
        if self._cfgs.pretraining:
            self._logger.info("Pre-training Mode On...")
        # xploit part
        self._agent = hydra.utils.instantiate(self._cfgs.agent)
        ## TODO: build encoder
        encoder = hydra.utils.instantiate(self._cfgs.encoder).to(self._device)
        ## TODO: build storage
        self._replay_storage = hydra.utils.instantiate(self._cfgs.storage)

        # xplore part
        ## TODO: get distribution
        if "Noise" in self._cfgs.distribution._target_:
            dist = hydra.utils.instantiate(self._cfgs.distribution)
        else:
            dist = hydra.utils.get_class(self._cfgs.distribution._target_)
        ## TODO: get augmentation
        aug = hydra.utils.instantiate(self._cfgs.augmentation).to(self._device) if self._cfgs.use_aug else None
        ## TODO: get intrinsic reward
        irs = hydra.utils.instantiate(self._cfgs.reward) if self._cfgs.use_irs else None

        # TODO: Integrate agent and modules
        self._agent.integrate(encoder=encoder, dist=dist, aug=aug, irs=irs)
        # TODO: load initial parameters
        if self._cfgs.init_model_path is not None:
            self._logger.info(f"Loading Initial Parameters from {self._cfgs.init_model_path}")
            self._agent.load(self._cfgs.init_model_path)

        self._num_init_steps = self._cfgs.num_init_steps

        # debug
        self._logger.debug("Check Accomplished. Start Training...")

    def train(self) -> None:
        """Training function."""
        episode_step, episode_reward = 0, 0
        obs, info = self._train_env.reset(seed=self._seed)
        metrics = None

        while self._global_step <= self._num_train_steps:
            # try to test
            if (self._global_step % self._test_every_steps) == 0 and (self._test_env is not None):
                test_metrics = self.test()
                self._logger.test(msg=test_metrics)

            # sample actions
            with th.no_grad(), eval_mode(self._agent):
                action = self._agent.act(obs, training=True, step=self._global_step)
                # TODO: Initial exploration
                if self._global_step < self._num_init_steps:
                    action.uniform_(-1.0, 1.0)
            next_obs, reward, terminated, truncated, info = self._train_env.step(action)
            episode_reward += reward[0].cpu().numpy()
            episode_step += 1
            self._global_step += 1

            # save transition
            self._replay_storage.add(
                obs[0].cpu().numpy(),
                action[0].cpu().numpy(),
                np.zeros_like(reward[0].cpu().numpy())
                if self._cfgs.pretraining
                else reward[0].cpu().numpy(),  # pre-training mode
                terminated[0].cpu().numpy(),
                info,
                next_obs[0].cpu().numpy(),
            )

            # update agent
            if self._global_step >= self._num_init_steps:
                metrics = self._agent.update(self._replay_storage, step=self._global_step)
                # try to update storage
                self._replay_storage.update(metrics)

            # done
            if terminated or truncated:
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
                    self._logger.train(msg=train_metrics)

                obs, info = self._train_env.reset(seed=self._seed)
                self._global_episode += 1
                episode_step, episode_reward = 0, 0
                continue

            obs = next_obs

        # save model
        self._logger.info("Training Accomplished!")
        self.save()

    def test(self) -> Dict[str, float]:
        """Testing function."""
        step, episode, total_reward = 0, 0, 0
        obs, info = self._test_env.reset(seed=self._seed)

        while episode <= self._num_test_episodes:
            with th.no_grad(), eval_mode(self._agent):
                action = self._agent.act(obs, training=False, step=self._global_step)

            next_obs, reward, terminated, truncated, info = self._test_env.step(action)
            total_reward += reward[0].cpu().numpy()
            step += 1

            if terminated or truncated:
                obs, info = self._test_env.reset(seed=self._seed)
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

    def save(self) -> None:
        """Save the trained model."""
        save_dir = Path.cwd() / "pretrained" if self._cfgs.pretraining else Path.cwd() / "model"
        save_dir.mkdir(exist_ok=True)
        self._agent.save(path=save_dir)
        self._logger.info(f"Model saved at: {save_dir}")
