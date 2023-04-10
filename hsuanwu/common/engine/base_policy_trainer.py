import os

os.environ["HYDRA_FULL_ERROR"] = "1"
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import open_dict, OmegaConf

from hsuanwu.common.engine.checker import cfgs_checker
from hsuanwu.common.logger import Logger, DEBUG, INFO
from hsuanwu.common.timer import Timer
from hsuanwu.common.typing import ABC, DictConfig, Env, abstractmethod, Tuple, Tensor, Space, Dict
from hsuanwu.xploit.learner import ALL_DEFAULT_CFGS


_DEFAULT_CFGS = {
    # Mandatory parameters
    ## TODO: Train setup
    'device': 'cpu',
    'seed': 1,
    'num_train_steps': 10000,
    ## TODO: Test setup
    "test_every_steps": 5000,
    "num_test_episodes": 10,
}


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
        # preprocess the cfgs
        processed_cfgs = self._process_cfgs(cfgs)
        cfgs_checker(logger=self._logger, cfgs=processed_cfgs)
        self._cfgs = self._set_class_path(processed_cfgs)
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
    
    def _remake_observation_and_action_space(self, observation_space: Space, action_space: Space) -> Tuple[Dict]:
        """Transform the original 'Box' space into Hydra supported type.

        Args:
            observation_space (Space): The observation space.
            action_space (Space): The action space.
        
        Returns:
            Processed spaces.
        """
        new_observation_space = {"shape": observation_space.shape}

        if action_space.__class__.__name__ == "Discrete":
            n = int(action_space.n)
            new_action_space = {"shape": (n, ), 
                                "type": "Discrete", 
                                "range": [0, n - 1]}
        elif action_space.__class__.__name__ == "Box":
            low, high = float(action_space.low[0]), float(action_space.high[0])
            new_action_space = {"shape": action_space.shape, 
                                "type": "Box", 
                                "range": [low, high]}
        else:
            raise NotImplementedError("Unsupported action type!")
        
        return new_observation_space, new_action_space



    def _process_cfgs(self, cfgs: DictConfig) -> DictConfig:
        """Preprocess the configs.

        Args:
            cfgs (DictConfig): Dict config for configuring RL algorithms.

        Returns:
            Processed configs.
        """
        new_cfgs = OmegaConf.create(_DEFAULT_CFGS)
        try:
            cfgs.learner.name
        except:
            raise ValueError(f"The learner name must be specified!")

        if cfgs.learner.name not in ALL_DEFAULT_CFGS.keys():
            raise NotImplementedError(f'Unsupported learner {cfgs.learner.name}, see https://docs.hsuanwu.dev/.')
        
        # TODO: try to load common configs
        for key in _DEFAULT_CFGS.keys():
            try:
                new_cfgs[key] = cfgs[key]
            except:
                pass
        
        # TODO: load the default configs of learner
        learner_default_cfgs = ALL_DEFAULT_CFGS[cfgs.learner.name]
        new_cfgs.merge_with(learner_default_cfgs)

        # TODO: try to load self-defined configs
        for part in ['encoder', 'learner', 'storage', 'distribution', 'augmentation', 'reward']:
            if part == 'augmentation' and not new_cfgs.use_aug: # don't use observation augmentation
                continue
            if part == 'reward' and not new_cfgs.use_irs: # don't use intrinsic reward
                continue

            for key in new_cfgs[part].keys():
                try:
                    new_cfgs[part][key] = cfgs[part][key]
                except:
                    pass
        
        # TODO: replace 'name' with '_target_' to use 'hydra.utils.instantiate'
        for part in ['encoder', 'learner', 'storage', 'distribution', 'augmentation', 'reward']:
            new_cfgs[part]['_target_'] = new_cfgs[part]['name']
            new_cfgs[part].pop('name')

        # TODO: remake observation and action sapce
        observation_space, action_space = self._remake_observation_and_action_space(
            self._train_env.observation_space, self._train_env.action_space)
        new_cfgs.observation_space = observation_space
        new_cfgs.action_space = action_space

        # TODO: fill parameters for encoder, learner, and storage
        ## for encoder
        if new_cfgs.encoder._target_ == 'IdentityEncoder':
            new_cfgs.encoder.feature_dim = observation_space['shape'][0]

        new_cfgs.encoder.observation_space = observation_space
        new_cfgs.learner.observation_space = observation_space
        new_cfgs.learner.action_space = action_space
        new_cfgs.learner.device = new_cfgs.device
        new_cfgs.learner.feature_dim = new_cfgs.encoder.feature_dim

        ## for storage
        if "Rollout" in new_cfgs.storage._target_:
            new_cfgs.storage.device = new_cfgs.device
            new_cfgs.storage.obs_shape = observation_space['shape']
            new_cfgs.storage.action_shape = action_space['shape']
            new_cfgs.storage.action_type = action_space['type']
            new_cfgs.storage.num_steps = new_cfgs.num_steps
            new_cfgs.storage.num_envs = self._train_env.num_envs

        if "Replay" in new_cfgs.storage._target_ and "NStep" not in new_cfgs.storage._target_:
            new_cfgs.storage.device = new_cfgs.device
            new_cfgs.storage.obs_shape = observation_space['shape']
            new_cfgs.storage.action_shape = action_space['shape']
            new_cfgs.storage.action_type = action_space['type']
        
        if "Distributed" in new_cfgs.storage._target_:
            new_cfgs.storage.device = new_cfgs.device
            new_cfgs.storage.obs_shape = observation_space['shape']
            new_cfgs.storage.action_shape = action_space['shape']
            new_cfgs.storage.num_steps = new_cfgs.num_steps

        ## for reward
        if new_cfgs.use_irs:
            new_cfgs.reward.device = new_cfgs.device
            new_cfgs.reward.obs_shape = observation_space["shape"]
            new_cfgs.reward.action_shape = action_space["shape"]

        return new_cfgs

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

    # @abstractmethod
    # def act(self, obs: Tensor, training: bool = True, step: int = 0) -> Tuple[Tensor]:
    #     """Sample actions based on observations.

    #     Args:
    #         obs: Observations.
    #         training: training mode, True or False.
    #         step: Global training step.

    #     Returns:
    #         Sampled actions.
    #     """

    # @abstractmethod
    # def train(self) -> None:
    #     """Training function."""

    # @abstractmethod
    # def test(self) -> None:
    #     """Testing function."""

    # @abstractmethod
    # def save(self) -> None:
    #     """Save the trained model."""
