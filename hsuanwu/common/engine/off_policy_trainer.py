from pathlib import Path

import hydra
import torch

from hsuanwu.common.typing import *
from hsuanwu.common.logger import *
from hsuanwu.common.timer import Timer

class OffPolicyTrainer:
    """Trainer for off-policy algorithms.
    
    Args:
        train_env: A Gym-like environment for training.
        test_env: A Gym-like environment for testing.
        cfgs: Dict config for configuring RL algorithms.

    Returns:
        Off-policy trainer instance.
    """
    def __init__(self,
                 train_env: Env,
                 test_env: Env,
                 cfgs: DictConfig) -> None:
        # setup
        self._cfgs = cfgs
        self._train_env = train_env
        self._test_env = test_env
        self._logger = Logger(log_dir=Path.cwd())
        self._timer = Timer()

        self._logger.log(INFO, 'Invoking Hsuanwu Engine...')
        # debug
        self._logger.log(DEBUG, 'Checking Module Compatibility...')

        # remake observation and action sapce
        cfgs.observation_space = {'shape': train_env.observation_space.shape}
        if cfgs.action_type == 'cont':
            cfgs.action_space = {'shape': train_env.action_space.shape}
        elif cfgs.action_type == 'dis':
            cfgs.action_space = {'shape': train_env.action_space.n}
        self._device = torch.device(cfgs.device)
        
        # xploit part
        self._learner = hydra.utils.instantiate(cfgs.learner)
        self._learner.encoder = hydra.utils.instantiate(cfgs.encoder).to(self._device)
        self._learner.encoder_opt = torch.optim.Adam(
            self._learner.encoder.parameters(), lr=cfgs.learner.lr)
        self._replay_buffer = hydra.utils.instantiate(cfgs.buffer)

        # xplore part
        self._learner.dist = hydra.utils.get_class(cfgs.distribution._target_)
        if cfgs.use_aug and cfgs.augmentation:
            self._learner.aug = hydra.utils.instantiate(cfgs.augmentation).to(self._device)
        if cfgs.use_irs:
            self._learner.reward = hydra.utils.instantiate(cfgs.reward)

        # make data loader        
        self._replay_loader = torch.utils.data.DataLoader(self._replay_buffer,
                                                  batch_size=cfgs.batch_size,
                                                  num_workers=cfgs.num_workers,
                                                  pin_memory=cfgs.pin_memory)
        self._replay_iter = None

        # training track
        self._global_step = 0
        self._global_episode = 0
        self._global_frame = 0
        self._train_unitl_step = cfgs.num_train_frames // cfgs.action_repeat
        self._seed_until_step = cfgs.num_seed_frames // cfgs.action_repeat
        self._test_every_steps = cfgs.test_every_frames // cfgs.action_repeat

        # debug
        self._logger.log(DEBUG, 'Check Accomplished. Start Training...')
    
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self._global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self._replay_loader)
        return self._replay_iter

    def train(self):
        episode_step, episode_reward = 0, 0
        obs = self._train_env.reset()
        metrics = None

        while self._global_step <= self._train_unitl_step:
            # try to test
            if self._global_step % self._test_every_steps == 0:
               test_metrics = self.test()
               self._logger.log(level=TEST, msg=test_metrics)
            
            # sample actions
            with torch.no_grad():
                action = self._learner.act(obs, training=True, step=self._global_step)
            next_obs, reward, done, info = self._train_env.step(action)
            episode_reward += reward
            episode_step += 1
            self._global_step += 1
            self._global_frame += 1 * self._cfgs.action_repeat

            # save transition
            self._replay_buffer.add(obs, action, reward, done)

            # update agent
            if self._global_step >= self._seed_until_step:
                metrics = self._learner.update(self.replay_iter, step=self._global_step)
            
            # done
            if done:
                episode_time, total_time = self._timer.reset()
                episode_frame = episode_step * self._cfgs.action_repeat

                if metrics is not None:
                    train_metrics = {
                        'frame': self._global_frame,
                        'step': self._global_step,
                        'episode': self._global_episode,
                        'episode_length': episode_frame,
                        'episode_reward': episode_reward,
                        'fps': episode_frame / episode_time,
                        'total_time': total_time,
                    }
                    self._logger.log(level=TRAIN, msg=train_metrics)

                obs = self._train_env.reset()
                self._global_episode += 1
                episode_step, episode_reward = 0, 0
                continue
            
            obs = next_obs

    def test(self):
        step, episode, total_reward = 0, 0, 0
        obs = self._test_env.reset()

        while episode <= self._cfgs.num_test_episodes:
            with torch.no_grad():
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
            'frame': self._global_frame,
            'step': self._global_step,
            'episode': self._global_episode,
            'episode_length': step * self._cfgs.action_repeat / episode,
            'episode_reward': total_reward / episode,
            'total_time': self._timer.total_time()
        }