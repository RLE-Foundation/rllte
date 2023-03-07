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
        self._logger = Logger(log_dir=cfgs.log_dir)
        self._timer = Timer()

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
        self._replay_loader = torch.utils.data.DataLoader(self.buffer,
                                                  batch_size=cfgs.batch_size,
                                                  num_workers=cfgs.num_workers,
                                                  pin_memory=cfgs.pin_memory)
        self._replay_iter = None

        # training track
        self._global_step = 0
        self._global_episode = 0
        self._train_unitl_step = cfgs.num_train_frames // cfgs.action_repeat
        self._seed_until_step = cfgs.num_seed_frames // cfgs.action_repeat
        self._test_every_steps = cfgs.eval_every_frames // cfgs.action_repeat
    
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self._replay_loader)
        return self._replay_iter

    def train(self):
        episode_step, episode_reward = 0, 0
        obs = self._train_env.reset()

        while self._global_step <= self._train_unitl_step:
            # try to test
            if self._global_step % self._test_every_steps:
               test_metrics = self.test()
               self._logger.log(level=TEST, msg=test_metrics)

    def test(self):
        pass