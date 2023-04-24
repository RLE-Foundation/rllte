import os
import numpy as np
import omegaconf
import torch as th
import gymnasium as gym
import random
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pathlib import Path
from omegaconf import OmegaConf

from hsuanwu.common.logger import Logger
from hsuanwu.common.timer import Timer
from hsuanwu.xploit.agent import ALL_DEFAULT_CFGS, ALL_MATCH_KEYS

os.environ["HYDRA_FULL_ERROR"] = "1"

_DEFAULT_CFGS = {
    # Mandatory parameters
    ## TODO: Train setup
    "device": "cpu",
    "seed": 1,
    "num_train_steps": 100000,
    "num_init_steps": 2000,  # only for off-policy algorithms
    "pretraining": False,
    ## TODO: Test setup
    "test_every_steps": 5000,  # only for off-policy algorithms
    "test_every_episodes": 10,  # only for on-policy algorithms
    "num_test_episodes": 10,
    ## TODO: xploit part
    "encoder": {
        "name": None,
    },
    "agent": {
        "name": None,
    },
    "storage": {
        "name": None,
    },
    ## TODO: xplore part
    "distribution": {"name": None},
    "augmentation": {"name": None},
    "reward": {"name": None},
}


class BasePolicyTrainer(ABC):
    """Base class of policy trainer.

    Args:
        cfgs (DictConfig): Dict config for configuring RL algorithms.
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.

    Returns:
        Base policy trainer instance.
    """

    def __init__(self, cfgs: omegaconf.DictConfig, train_env: gym.Env, test_env: gym.Env = None) -> None:
        # basic setup
        self._train_env = train_env
        self._test_env = test_env
        self._work_dir = Path.cwd()
        self._logger = Logger(log_dir=self._work_dir)
        self._timer = Timer()
        self._device = th.device(cfgs.device)
        # set seed
        self._seed = cfgs.seed
        th.manual_seed(seed=cfgs.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(cfgs.seed)
        np.random.seed(cfgs.seed)
        random.seed(cfgs.seed)
        # debug
        try:
            self._logger.info(f"Experiment: {cfgs.experiment}")
        except:
            self._logger.info("Experiment: Default")
        self._logger.info("Invoking Hsuanwu Engine...")
        # preprocess the cfgs
        processed_cfgs = self._process_cfgs(cfgs)
        # check the cfgs
        self._check_cfgs(cfgs=processed_cfgs)
        self._cfgs = self._set_class_path(processed_cfgs)
        # training track
        self._num_train_steps = self._cfgs.num_train_steps
        self._num_test_episodes = self._cfgs.num_test_episodes
        self._test_every_steps = self._cfgs.test_every_steps  # only for off-policy algorithms
        self._test_every_episodes = self._cfgs.test_every_episodes  # only for on-policy algorithms
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

    def _process_cfgs(self, cfgs: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Preprocess the configs.

        Args:
            cfgs (DictConfig): Dict config for configuring RL algorithms.

        Returns:
            Processed configs.
        """
        new_cfgs = OmegaConf.create(_DEFAULT_CFGS)
        # TODO: load the default configs of agent
        agent_default_cfgs = ALL_DEFAULT_CFGS[cfgs.agent.name]

        for key in agent_default_cfgs.keys():
            new_cfgs[key] = agent_default_cfgs[key]

        # TODO: try to load self-defined configs
        for part in cfgs.keys():
            if part not in [
                "encoder",
                "agent",
                "storage",
                "distribution",
                "augmentation",
                "reward",
            ]:
                new_cfgs[part] = cfgs[part]

        for part in cfgs.keys():
            if part in [
                "encoder",
                "agent",
                "storage",
                "distribution",
                "augmentation",
                "reward",
            ]:
                for key in cfgs[part].keys():
                    new_cfgs[part][key] = cfgs[part][key]

        ## TODO: replace 'name' with '_target_' to use 'hydra.utils.instantiate'
        for part in new_cfgs.keys():
            if part in [
                "encoder",
                "agent",
                "storage",
                "distribution",
                "augmentation",
                "reward",
            ]:
                new_cfgs[part]["_target_"] = new_cfgs[part]["name"]
                new_cfgs[part].pop("name")

        ## TODO: set flag for 'augmentation' and 'reward'
        if new_cfgs.augmentation._target_ is not None:
            new_cfgs.use_aug = True
        else:
            new_cfgs.use_aug = False

        if new_cfgs.reward._target_ is not None:
            new_cfgs.use_irs = True
        else:
            new_cfgs.use_irs = False

        observation_space, action_space = (
            self._train_env.observation_space,
            self._train_env.action_space,
        )

        new_cfgs.num_envs = self._train_env.num_envs
        new_cfgs.observation_space = observation_space
        new_cfgs.action_space = action_space

        # TODO: fill parameters for encoder, agent, and storage
        ## for encoder
        if new_cfgs.encoder._target_ == "IdentityEncoder":
            new_cfgs.encoder.feature_dim = observation_space["shape"][0]

        new_cfgs.encoder.observation_space = observation_space
        new_cfgs.agent.observation_space = observation_space
        new_cfgs.agent.action_space = action_space
        new_cfgs.agent.device = new_cfgs.device
        new_cfgs.agent.feature_dim = new_cfgs.encoder.feature_dim

        ## for storage
        new_cfgs.storage.observation_space = observation_space
        new_cfgs.storage.action_space = action_space
        new_cfgs.storage.device = new_cfgs.device

        if "Rollout" in new_cfgs.storage._target_:
            new_cfgs.storage.num_steps = new_cfgs.num_steps
            new_cfgs.storage.num_envs = new_cfgs.num_envs

        if "Distributed" in new_cfgs.storage._target_:
            new_cfgs.storage.num_steps = new_cfgs.num_steps

        ## for reward
        if new_cfgs.reward._target_ is not None:
            new_cfgs.reward.device = new_cfgs.device
            new_cfgs.reward.observation_space = observation_space
            new_cfgs.reward.action_space = action_space

        return new_cfgs

    def _check_cfgs(self, cfgs: omegaconf.DictConfig) -> None:
        """Check the compatibility of selected modules.
        Args:
            cfgs (DictConfig): Dict Config.

        Returns:
            None.
        """
        self._logger.debug("Checking the Compatibility of Modules...")

        # xploit part
        self._logger.debug(f"Selected Encoder: {cfgs.encoder._target_}")
        self._logger.debug(f"Selected Agent: {cfgs.agent._target_}")
        # Check the compatibility
        assert (
            cfgs.storage._target_ in ALL_MATCH_KEYS[cfgs.agent._target_]["storage"]
        ), f"{cfgs.storage._target_} is incompatible with {cfgs.agent._target_}! See https://docs.hsuanwu.dev/."
        self._logger.debug(f"Selected Storage: {cfgs.storage._target_}")

        assert (
            cfgs.distribution._target_ in ALL_MATCH_KEYS[cfgs.agent._target_]["distribution"]
        ), f"{cfgs.distribution._target_} is incompatible with {cfgs.agent._target_}! See https://docs.hsuanwu.dev/."
        self._logger.debug(f"Selected Distribution: {cfgs.distribution._target_}")

        if cfgs.augmentation._target_ is not None:
            self._logger.debug(f"Use Augmentation: {cfgs.use_aug}, {cfgs.augmentation._target_}")
        else:
            self._logger.debug(f"Use Augmentation: {cfgs.use_aug}")

        if cfgs.pretraining:
            assert (
                cfgs.reward._target_ is not None
            ), "When the pre-training mode is turned on, an intrinsic reward must be specified!"

        if cfgs.reward._target_ is not None:
            self._logger.debug(f"Use Intrinsic Reward: {cfgs.use_irs}, {cfgs.reward._target_}")
        else:
            self._logger.debug(f"Use Intrinsic Reward: {cfgs.use_irs}")

    def _set_class_path(self, cfgs: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Set the class path for each module.

        Args:
            cfgs (DictConfig): Dict config for configuring RL algorithms.

        Returns:
            Processed configs.
        """
        cfgs.agent._target_ = "hsuanwu.xploit." + "agent." + cfgs.agent._target_
        cfgs.encoder._target_ = "hsuanwu.xploit." + "encoder." + cfgs.encoder._target_
        cfgs.storage._target_ = "hsuanwu.xploit." + "storage." + cfgs.storage._target_

        cfgs.distribution._target_ = "hsuanwu.xplore." + "distribution." + cfgs.distribution._target_
        if cfgs.augmentation._target_ is not None:
            cfgs.augmentation._target_ = "hsuanwu.xplore." + "augmentation." + cfgs.augmentation._target_
        if cfgs.reward._target_ is not None:
            cfgs.reward._target_ = "hsuanwu.xplore." + "reward." + cfgs.reward._target_

        return cfgs

    @abstractmethod
    def train(self) -> Optional[Dict[str, float]]:
        """Training function."""

    @abstractmethod
    def test(self) -> Optional[Dict[str, float]]:
        """Testing function."""

    @abstractmethod
    def save(self) -> None:
        """Save the trained model."""
