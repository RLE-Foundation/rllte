import hydra
import torch
from pathlib import Path

from hsuanwu.common.engine import BasePolicyTrainer, utils
from hsuanwu.common.logger import Logger, INFO, DEBUG, TEST, TRAIN
from hsuanwu.common.typing import DictConfig, Env, Iterable, Tensor, Tuple
from hsuanwu.xploit.storage.utils import worker_init_fn


class OffPolicyTrainer(BasePolicyTrainer):
    """Trainer for off-policy algorithms.

    Args:
        cfgs (DictConfig): Dict config for configuring RL algorithms.
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.

    Returns:
        Off-policy trainer instance.
    """

    def __init__(self, cfgs: DictConfig, train_env: Env, test_env: Env = None) -> None:
        super().__init__(cfgs, train_env, test_env)
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
        self._replay_storage = hydra.utils.instantiate(self._cfgs.storage)

        # xplore part
        # TODO: get distribution
        if "Noise" in self._cfgs.distribution._target_:
            dist = hydra.utils.instantiate(self._cfgs.distribution)
        else:
            dist = hydra.utils.get_class(self._cfgs.distribution._target_)
        self._learner.dist = dist
        self._learner.actor.dist = dist
        # TODO: get augmentation
        if self._cfgs.use_aug:
            self._learner.aug = hydra.utils.instantiate(self._cfgs.augmentation).to(
                self._device
            )
        # TODO: get intrinsic reward
        if self._cfgs.use_irs:
            self._learner.irs = hydra.utils.instantiate(self._cfgs.reward)

        # make data loader
        if "NStepReplayStorage" in self._cfgs.storage._target_:
            self._replay_loader = torch.utils.data.DataLoader(
                self._replay_storage,
                batch_size=self._replay_storage.get_batch_size,
                num_workers=self._replay_storage.get_num_workers,
                pin_memory=self._replay_storage.get_pin_memory,
                worker_init_fn=worker_init_fn,
            )
            self._replay_iter = None
            self._use_nstep_replay_storage = True
        else:
            self._use_nstep_replay_storage = False

        self._num_init_steps = self._cfgs.num_init_steps

        # debug
        self._logger.log(DEBUG, "Check Accomplished. Start Training...")

    @property
    def replay_iter(self) -> Iterable:
        """Create iterable dataloader."""
        if self._replay_iter is None:
            self._replay_iter = iter(self._replay_loader)
        return self._replay_iter

    def act(self, obs: Tensor, training: bool = True, step: int = 0) -> Tuple[Tensor]:
        """Sample actions based on observations.

        Args:
            obs (Tensor): Observations.
            training (bool): training mode, True or False.
            step (int): Global training step.

        Returns:
            Sampled actions.
        """
        # sample actions
        encoded_obs = self._learner.encoder(obs)
        dist = self._learner.actor.get_action(obs=encoded_obs, step=self._global_step)

        if not training:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self._num_init_steps:
                action.uniform_(-1.0, 1.0)
        return action.clamp(*self._cfgs.action_space['range'])

    def train(self) -> None:
        """Training function."""
        episode_step, episode_reward = 0, 0
        obs, info = self._train_env.reset(seed=self._seed)
        metrics = None

        while self._global_step <= self._num_train_steps:
            # try to test
            if (self._global_step % self._test_every_steps) == 0 and (self._test_env is not None):
                test_metrics = self.test()
                self._logger.log(level=TEST, msg=test_metrics)

            # sample actions
            with torch.no_grad(), utils.eval_mode(self._learner):
                action = self.act(obs, training=True, step=self._global_step)
            next_obs, reward, terminated, truncated, info = self._train_env.step(action)
            episode_reward += reward[0].cpu().numpy()
            episode_step += 1
            self._global_step += 1

            # save transition
            self._replay_storage.add(
                obs[0].cpu().numpy(),
                action[0].cpu().numpy(),
                reward[0].cpu().numpy(),
                terminated[0].cpu().numpy(),
                info,
                next_obs[0].cpu().numpy(),
            )

            # update agent
            if self._global_step >= self._num_init_steps:
                if self._use_nstep_replay_storage:
                    # TODO: for NStepReplayStorage
                    metrics = self._learner.update(
                        self.replay_iter, step=self._global_step
                    )
                else:
                    metrics = self._learner.update(
                        self._replay_storage, step=self._global_step
                    )

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
                    self._logger.log(level=TRAIN, msg=train_metrics)

                obs, info = self._train_env.reset(seed=self._seed)
                self._global_episode += 1
                episode_step, episode_reward = 0, 0
                continue

            obs = next_obs

        # save model
        self.save()

    def test(self) -> None:
        """Testing function."""
        step, episode, total_reward = 0, 0, 0
        obs, info = self._test_env.reset(seed=self._seed)

        while episode <= self._num_test_episodes:
            with torch.no_grad(), utils.eval_mode(self._learner):
                action = self.act(obs, training=False, step=self._global_step)

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
        save_dir = Path.cwd() / "model"
        save_dir.mkdir(exist_ok=True)
        torch.save(self._learner.encoder, save_dir / "encoder.pth")
        torch.save(self._learner.actor, save_dir / "actor.pth")
        torch.save(self._learner.critic, save_dir / "critic.pth")
