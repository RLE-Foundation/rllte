from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch

from hsuanwu.common.engine import BasePolicyTrainer, utils
from hsuanwu.common.typing import Dict, DictConfig, Env, Tensor, Tuple


class OnPolicyTrainer(BasePolicyTrainer):
    """Trainer for on-policy algorithms.

    Args:
        cfgs (DictConfig): Dict config for configuring RL algorithms.
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.

    Returns:
        On-policy trainer instance.
    """

    def __init__(self, cfgs: DictConfig, train_env: Env, test_env: Env = None) -> None:
        super().__init__(cfgs, train_env, test_env)
        self._logger.info(f"Deploying OnPolicyTrainer...")
        # xploit part
        self._learner = hydra.utils.instantiate(self._cfgs.learner)
        # TODO: build encoder
        self._learner.encoder = hydra.utils.instantiate(self._cfgs.encoder).to(
            self._device
        )
        self._learner.encoder.train()
        self._learner.encoder_opt = torch.optim.Adam(
            self._learner.encoder.parameters(),
            lr=self._learner.lr,
            eps=self._learner.eps,
        )
        # TODO: build storage
        self._rollout_storage = hydra.utils.instantiate(self._cfgs.storage)

        # xplore part
        # TODO: get distribution
        dist = hydra.utils.get_class(self._cfgs.distribution._target_)
        self._learner.dist = dist
        self._learner.ac.dist = dist
        # TODO: get augmentation
        if self._cfgs.use_aug:
            self._learner.aug = hydra.utils.instantiate(self._cfgs.augmentation).to(
                self._device
            )
        # TODO: get intrinsic reward
        if self._cfgs.use_irs:
            self._learner.irs = hydra.utils.instantiate(self._cfgs.reward)

        self._num_steps = self._cfgs.num_steps
        self._num_envs = self._cfgs.num_envs

        # debug
        self._logger.debug("Check Accomplished. Start Training...")

    def act(self, obs: Tensor, training: bool = True, step: int = 0) -> Tuple[Tensor]:
        """Sample actions based on observations.

        Args:
            obs: Observations.
            training: training mode, True or False.
            step: Global training step.

        Returns:
            Sampled actions.
        """
        encoded_obs = self._learner.encoder(obs)

        if training:
            actions, values, log_probs, entropy = self._learner.ac.get_action_and_value(
                obs=encoded_obs
            )
            return actions, values, log_probs, entropy
        else:
            actions = self._learner.ac.get_action(obs=encoded_obs)
            return actions

    def train(self) -> None:
        """Training function."""
        episode_rewards = deque(maxlen=10)
        episode_steps = deque(maxlen=10)
        obs, info = self._train_env.reset(seed=self._seed)
        metrics = None
        # Number of updates
        num_updates = self._num_train_steps // self._num_envs // self._num_steps

        for update in range(num_updates):
            # try to test
            if (update % self._test_every_episodes) == 0 and (
                self._test_env is not None
            ):
                test_metrics = self.test()
                self._logger.test(msg=test_metrics)

            for step in range(self._num_steps):
                # sample actions
                with torch.no_grad(), utils.eval_mode(self._learner):
                    actions, values, log_probs, entropy = self.act(
                        obs, training=True, step=self._global_step
                    )
                (
                    next_obs,
                    rewards,
                    terminateds,
                    truncateds,
                    infos,
                ) = self._train_env.step(actions)

                if "episode" in infos:
                    indices = np.nonzero(infos["episode"]["r"])
                    episode_rewards.extend(infos["episode"]["r"][indices].tolist())
                    episode_steps.extend(infos["episode"]["l"][indices].tolist())

                # add transitions
                self._rollout_storage.add(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    terminateds=terminateds,
                    truncateds=truncateds,
                    log_probs=log_probs,
                    values=values,
                )

                obs = next_obs

            # get the value estimation of the last step
            with torch.no_grad():
                last_values = self._learner.get_value(next_obs).detach()

            # perform return and advantage estimation
            self._rollout_storage.compute_returns_and_advantages(last_values)

            # policy update
            metrics = self._learner.update(
                self._rollout_storage, episode=self._global_episode
            )

            # reset buffer
            self._rollout_storage.reset()

            self._global_episode += 1
            self._global_step += self._num_envs * self._num_steps
            episode_time, total_time = self._timer.reset()

            if len(episode_rewards) > 1:
                train_metrics = {
                    "step": self._global_step,
                    "episode": self._global_episode,
                    "episode_length": np.mean(episode_steps),
                    "episode_reward": np.mean(episode_rewards),
                    "fps": self._num_steps * self._num_envs / episode_time,
                    "total_time": total_time,
                }
                self._logger.train(msg=train_metrics)

        # save model
        self._logger.info("Training Accomplished!")
        self.save()

    def test(self) -> Dict:
        """Testing function."""
        obs, info = self._test_env.reset(seed=self._seed)
        episode_rewards = list()
        episode_steps = list()

        while len(episode_rewards) < self._num_test_episodes:
            with torch.no_grad(), utils.eval_mode(self._learner):
                actions = self.act(obs, training=False, step=self._global_step)
            obs, rewards, terminateds, truncateds, infos = self._test_env.step(actions)

            if "episode" in infos:
                indices = np.nonzero(infos["episode"]["r"])
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
        save_dir = Path.cwd() / "model"
        save_dir.mkdir(exist_ok=True)
        torch.save(self._learner.encoder, save_dir / "encoder.pth")
        torch.save(self._learner.ac, save_dir / "actor_critic.pth")
