import os
os.environ['HYDRA_FULL_ERROR'] = '1'
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import open_dict

from hsuanwu.common.engine.checker import cfgs_checker
from hsuanwu.common.logger import *
from hsuanwu.common.timer import Timer
from hsuanwu.common.typing import Env, DictConfig, ABC, abstractmethod


class BasePolicyTrainer(ABC):
    """Base class of policy trainer.

    Args:
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.
        cfgs (DictConfig): Dict config for configuring RL algorithms.

    Returns:
        Base policy trainer instance.
    """

    def __init__(self, train_env: Env, test_env: Env, cfgs: DictConfig) -> None:
        # basic setup
        self._train_env = train_env
        self._test_env = test_env
        self._work_dir = Path.cwd()
        self._logger = Logger(log_dir=self._work_dir)
        self._timer = Timer()
        self._device = torch.device(cfgs.device)
        # set seed
        self._seed = cfgs.seed
        torch.manual_seed(seed=cfgs.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfgs.seed)
        np.random.seed(cfgs.seed)
        random.seed(cfgs.seed)
        # debug
        self._logger.log(INFO, "Invoking Hsuanwu Engine...")
        cfgs_checker(logger=self._logger, cfgs=cfgs)
        # preprocess the cfgs
        self._cfgs = self._process_cfgs(cfgs)
        # training track
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self) -> int:
        """Get global training steps."""
        return self._global_step

    @property
    def global_episode(self) -> int:
        """Get global training episodes."""
        return self._global_episode

    def _process_cfgs(self, cfgs: DictConfig) -> DictConfig:
        """Preprocess the configs.

        Args:
            cfgs (DictConfig): Dict config for configuring RL algorithms.

        Returns:
            Processed configs.
        """
        # remake observation and action sapce
        obs_shape = self._train_env.observation_space.shape
        observation_space = {"shape": self._train_env.observation_space.shape}
        if self._train_env.action_space.__class__.__name__ == "Discrete":
            action_space = {"shape": (self._train_env.action_space.n,)}
            action_type = "dis"
            self._action_range = [0, self._train_env.action_space.n - 1]
        elif self._train_env.action_space.__class__.__name__ == "Box":
            action_space = {"shape": self._train_env.action_space.shape}
            action_type = "cont"
            self._action_range = [self._train_env.action_space.low[0], self._train_env.action_space.high[0]]
        else:
            raise NotImplementedError("Unsupported action type!")

        # set observation and action space for learner and encoder
        with open_dict(cfgs):
            # for encoder
            cfgs.encoder.observation_space = observation_space
            # for learner
            cfgs.learner.observation_space = observation_space
            cfgs.learner.action_space = action_space
            cfgs.learner.action_type = action_type
            cfgs.learner.device = cfgs.device
            cfgs.learner.feature_dim = cfgs.encoder.feature_dim

        # set observation and action shape for rollout storage.
        if "Rollout" in cfgs.storage._target_:
            with open_dict(cfgs):
                cfgs.storage.device = cfgs.device
                cfgs.storage.obs_shape = obs_shape
                cfgs.storage.action_shape = action_space["shape"]
                cfgs.storage.action_type = action_type
                cfgs.storage.num_steps = cfgs.num_steps
                cfgs.storage.num_envs = cfgs.num_envs

        # set observation and action shape for replay storage.
        if "Replay" in cfgs.storage._target_ and "NStep" not in cfgs.storage._target_:
            with open_dict(cfgs):
                cfgs.storage.device = cfgs.device
                cfgs.storage.obs_shape = obs_shape
                cfgs.storage.action_shape = action_space["shape"]
                cfgs.storage.action_type = action_type

        # xplore part
        if cfgs.use_irs:
            with open_dict(cfgs):
                cfgs.reward.obs_shape = observation_space["shape"]
                cfgs.reward.action_shape = action_space["shape"]
                cfgs.reward.action_type = action_type
                cfgs.reward.device = cfgs.device

        return self._set_class_path(cfgs)

    def _set_class_path(self, cfgs: DictConfig) -> DictConfig:
        """Set the class path for each module.

        Args:
            cfgs (DictConfig): Dict config for configuring RL algorithms.

        Returns:
            Processed configs.
        """
        cfgs.learner._target_ = "hsuanwu.xploit." + "learner." + cfgs.learner._target_
        cfgs.encoder._target_ = "hsuanwu.xploit." + "encoder." + cfgs.encoder._target_
        cfgs.storage._target_ = "hsuanwu.xploit." + "storage." + cfgs.storage._target_

        cfgs.distribution._target_ = (
            "hsuanwu.xplore." + "distribution." + cfgs.distribution._target_
        )
        if cfgs.use_aug:
            cfgs.augmentation._target_ = (
                "hsuanwu.xplore." + "augmentation." + cfgs.augmentation._target_
            )
        if cfgs.use_irs:
            cfgs.reward._target_ = "hsuanwu.xplore." + "reward." + cfgs.reward._target_

        return cfgs

    @abstractmethod
    def act(
        self, obs: Tensor, training: bool = True, step: int = 0
    ) -> Tuple[Any]:
        """Sample actions based on observations.

        Args:
            obs: Observations.
            training: training mode, True or False.
            step: Global training step.

        Returns:
            Sampled actions.
        """
    
    @abstractmethod
    def train(self) -> None:
        """Training function."""
    
    @abstractmethod
    def test(self) -> None:
        """Testing function."""

    @abstractmethod
    def save(self) -> None:
        """Save the trained model."""