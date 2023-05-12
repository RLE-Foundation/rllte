from collections import deque
from pathlib import Path
from typing import Dict

import gymnasium as gym
import hydra
import numpy as np
import omegaconf
import torch as th

from hsuanwu.common.engine.base_policy_trainer import BasePolicyTrainer
from hsuanwu.common.engine.utils import eval_mode


class OnPolicyTrainer(BasePolicyTrainer):
    """Trainer for on-policy algorithms.

    Args:
        cfgs (DictConfig): Dict config for configuring RL algorithms.
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.

    Returns:
        On-policy trainer instance.
    """

    def __init__(self, cfgs: omegaconf.DictConfig, train_env: gym.Env, test_env: gym.Env = None) -> None:
        super().__init__(cfgs, train_env, test_env)
        self._logger.info("Deploying OnPolicyTrainer...")
        # TODO: turn on the pretraining mode, no extrinsic rewards will be provided.
        if self._cfgs.pretraining:
            self._logger.info("Pre-training Mode On...")
        # xploit part
        self._agent = hydra.utils.instantiate(self._cfgs.agent)
        # TODO: build encoder
        encoder = hydra.utils.instantiate(self._cfgs.encoder).to(self._device)
        # TODO: build storage
        self._rollout_storage = hydra.utils.instantiate(self._cfgs.storage)

        # xplore part
        # TODO: get distribution
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

        self._num_steps = self._cfgs.num_steps
        self._num_envs = self._cfgs.num_envs

        # debug
        self._logger.debug("Check Accomplished. Start Training...")

    def train(self) -> None:
        """Training function."""
        episode_rewards = deque(maxlen=100)
        episode_steps = deque(maxlen=100)
        obs, info = self._train_env.reset(seed=self._seed)
        # Number of updates
        num_updates = self._num_train_steps // self._num_envs // self._num_steps

        for update in range(num_updates):
            # try to test
            if (update % self._test_every_episodes) == 0 and (self._test_env is not None):
                test_metrics = self.test()
                self._logger.test(msg=test_metrics)

            for _step in range(self._num_steps):
                # sample actions
                with th.no_grad(), eval_mode(self._agent):
                    agent_outputs = self._agent.act(obs, training=True, step=self._global_step)

                (
                    next_obs,
                    rewards,
                    terminateds,
                    truncateds,
                    infos,
                ) = self._train_env.step(agent_outputs["actions"].clamp(*self._action_range))

                agent_outputs.update(
                    {
                        "obs": obs,
                        "rewards": th.zeros_like(rewards, device=self._device)
                        if self._cfgs.pretraining
                        else rewards,  # pre-training mode
                        "terminateds": terminateds,
                        "truncateds": truncateds,
                        "next_obs": next_obs,
                    }
                )

                if "episode" in infos:
                    indices = np.nonzero(infos["episode"]["l"])
                    episode_rewards.extend(infos["episode"]["r"][indices].tolist())
                    episode_steps.extend(infos["episode"]["l"][indices].tolist())

                # add transitions
                self._rollout_storage.add(**agent_outputs)

                obs = next_obs

            # get the value estimation of the last step
            with th.no_grad():
                last_values = self._agent.get_value(next_obs).detach()

            # perform return and advantage estimation
            self._rollout_storage.compute_returns_and_advantages(last_values)

            # policy update
            self._agent.update(self._rollout_storage, episode=self._global_episode)

            # update and reset buffer
            self._rollout_storage.update()

            self._global_episode += 1
            self._global_step += self._num_envs * self._num_steps
            episode_time, total_time = self._timer.reset()

            if len(episode_rewards) > 1:
                train_metrics = {
                    "step": self._global_step,
                    "episode": self._global_episode * self._num_envs,
                    "episode_length": np.mean(episode_steps),
                    "episode_reward": np.mean(episode_rewards),
                    "fps": self._num_steps * self._num_envs / episode_time,
                    "total_time": total_time,
                }
                self._logger.train(msg=train_metrics)

        # save model
        self._logger.info("Training Accomplished!")
        self.save()
        self._train_env.close()
        if self._test_env is not None:
            self._test_env.close()

    def test(self) -> Dict[str, float]:
        """Testing function."""
        obs, info = self._test_env.reset(seed=self._seed)
        episode_rewards = list()
        episode_steps = list()

        while len(episode_rewards) < self._num_test_episodes:
            with th.no_grad(), eval_mode(self._agent):
                actions = self._agent.act(obs, training=False, step=self._global_step)
            obs, rewards, terminateds, truncateds, infos = self._test_env.step(actions.clamp(*self._action_range))

            if "episode" in infos:
                indices = np.nonzero(infos["episode"]["l"])
                episode_rewards.extend(infos["episode"]["r"][indices].tolist())
                episode_steps.extend(infos["episode"]["l"][indices].tolist())

        return {
            "step": self._global_step,
            "episode": self._global_episode,
            "episode_length": np.mean(episode_steps),
            "episode_reward": np.mean(episode_rewards),
            "total_time": self._timer.total_time(),
        }

    def save(self) -> None:
        """Save the trained model."""
        save_dir = Path.cwd() / "pretrained" if self._cfgs.pretraining else Path.cwd() / "model"
        save_dir.mkdir(exist_ok=True)
        self._agent.save(path=save_dir)
        self._logger.info(f"Model saved at: {save_dir}")
