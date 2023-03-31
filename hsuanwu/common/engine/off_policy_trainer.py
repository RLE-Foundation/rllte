import hydra
import torch

from hsuanwu.common.engine import BasePolicyTrainer, utils
from hsuanwu.common.logger import *
from hsuanwu.common.typing import Env, DictConfig
from hsuanwu.xploit.storage.utils import worker_init_fn


class OffPolicyTrainer(BasePolicyTrainer):
    """Trainer for off-policy algorithms.

    Args:
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.
        cfgs (DictConfig): Dict config for configuring RL algorithms.

    Returns:
        Off-policy trainer instance.
    """

    def __init__(self, train_env: Env, test_env: Env, cfgs: DictConfig) -> None:
        super().__init__(train_env, test_env, cfgs)
        # xploit part
        self._learner = hydra.utils.instantiate(self._cfgs.learner)
        # TODO: build encoder
        self._learner.encoder = hydra.utils.instantiate(self._cfgs.encoder).to(self._device)
        self._learner.encoder.train()
        self._learner.encoder_opt = torch.optim.Adam(
            self._learner.encoder.parameters(), lr=self._learner.lr, eps=self._learner.eps
        )
        # TODO: build storage
        self._replay_storage = hydra.utils.instantiate(self._cfgs.storage)

        # xplore part
        # TODO: get distribution
        dist = hydra.utils.get_class(self._cfgs.distribution._target_)
        self._learner.dist = dist
        self._learner.actor.dist = dist
        # TODO: get augmentation
        if self._cfgs.use_aug:
            self._learner.aug = hydra.utils.instantiate(self._cfgs.augmentation).to(self._device)
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
    
    def act(
        self, obs: Tensor, training: bool = True, step: int = 0
    ) -> Tuple[Tensor]:
        """Sample actions based on observations.

        Args:
            obs: Observations.
            training: training mode, True or False.
            step: Global training step.

        Returns:
            Sampled actions.
        """
        encoded_obs = self._learner.encoder(obs.unsqueeze(0))
        # sample actions
        # TODO: manual exploration noise control? (for continuous control task) \
        # See paper: https://openreview.net/forum?id=_SJ-_yyes8, Section 3.1.
        if self._cfgs.stddev_schedule:
            std = utils.schedule(self._cfgs.stddev_schedule, step)
        else:
            std = None
        dist = self._learner.actor(obs=encoded_obs, std=std)

        if not training:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self._num_init_steps:
                action.uniform_(-1.0, 1.0)

        return action

    def train(self) -> None:
        """Training function."""
        episode_step, episode_reward = 0, 0
        obs, info = self._train_env.reset(seed=self._seed)
        metrics = None

        while self._global_step <= self._num_train_steps:
            # try to test
            if self._global_step % self._test_every_steps == 0:
                test_metrics = self.test()
                self._logger.log(level=TEST, msg=test_metrics)

            # sample actions
            with torch.no_grad(), utils.eval_mode(self._learner):
                action = self.act(obs, training=True, step=self._global_step)
            next_obs, reward, terminated, truncated, info = self._train_env.step(action)
            episode_reward += reward
            episode_step += 1
            self._global_step += 1

            # save transition
            self._replay_storage.add(obs.cpu().numpy(), 
                                     action.cpu().numpy()[0], 
                                     reward.cpu().numpy(), 
                                     terminated.cpu().numpy(), 
                                     info, 
                                     next_obs.cpu().numpy())

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

    def test(self) -> None:
        """Testing function."""
        step, episode, total_reward = 0, 0, 0
        obs, info = self._test_env.reset(seed=self._seed)

        while episode <= self._cfgs.num_test_episodes:
            with torch.no_grad(), utils.eval_mode(self._learner):
                action = self.act(obs, training=False, step=self._global_step)

            next_obs, reward, terminated, truncated, info = self._test_env.step(action)
            total_reward += reward
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
