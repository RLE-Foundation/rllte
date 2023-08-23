# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import pynvml
import torch as th
th.set_float32_matmul_precision('high')

# try to load torch_npu
try:
    import torch_npu as torch_npu  # type: ignore
    if th.npu.is_available():
        NPU_AVAILABLE = True
except Exception:
    NPU_AVAILABLE = False

from rllte.common.prototype.base_augmentation import BaseAugmentation as Augmentation
from rllte.common.prototype.base_distribution import BaseDistribution as Distribution
from rllte.common.prototype.base_encoder import BaseEncoder as Encoder
from rllte.common.prototype.base_policy import BasePolicy as Policy
from rllte.common.prototype.base_reward import BaseIntrinsicRewardModule as IntrinsicRewardModule
from rllte.common.prototype.base_storage import BaseStorage as Storage
from rllte.common.logger import Logger
from rllte.common.preprocessing import process_observation_space, process_action_space
from rllte.common.timer import Timer
from rllte.common.utils import get_npu_name

NUMBER_OF_SPACES = 17


class BaseAgent(ABC):
    """Base class of the agent.

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.

    Returns:
        Base agent instance.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "auto",
        pretraining: bool = False,
    ) -> None:
        # change work dir
        self.tag = tag
        path = Path.cwd() / "logs" / tag / datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
        os.makedirs(path)
        os.chdir(path=path)

        # set logger and timer
        self.work_dir = Path.cwd()
        self.logger = Logger(log_dir=self.work_dir)
        self.timer = Timer()

        # set device and get device name
        if device == "auto":
            if NPU_AVAILABLE:
                device = "npu"
            else:
                device = "cuda" if th.cuda.is_available() else "cpu"
        self.device = th.device(device)
        ## get device name
        if "cuda" in device:
            try:
                device_id = int(device[-1])
            except Exception:
                device_id = 0
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.device_name = pynvml.nvmlDeviceGetName(handle)
        elif "npu" in device:
            self.device_name = f"HUAWEI Ascend {get_npu_name()}"
        else:
            self.device_name = "CPU"

        # env setup
        self.env = env
        self.eval_env = eval_env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs
        self.obs_shape = process_observation_space(env.observation_space)
        self.action_shape, self.action_dim, self.policy_action_dim, self.action_type = process_action_space(env.action_space)

        # set seed
        self.seed = seed
        th.manual_seed(seed=seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # placeholder for Encoder, Storage, Distribution, Augmentation, Reward
        self.encoder = None
        self.policy = None
        self.storage = None
        self.dist = None
        self.aug = None
        self.irs = None

        # training tracking
        self.pretraining = pretraining
        self.global_step = 0
        self.global_episode = 0

    def freeze(self, **kwargs) -> None:
        """Freeze the agent and get ready for training."""
        # freeze the structure of the agent
        self.policy.freeze(encoder=self.encoder, dist=self.dist)
        # torch compilation
        ## compliation will change the name of the policy into `OptimizedModule`
        self.policy_name = self.policy.__class__.__name__
        if kwargs.get("th_compile", False):
            self.policy = th.compile(self.policy)
        # to device
        self.policy.to(self.device)
        # set the training mode
        self.mode(training=True)
        # final check
        self.check()

        # load initial model parameters
        init_model_path = kwargs.get("init_model_path", None)
        if init_model_path is not None:
            self.logger.info(f"Loading Initial Parameters from {init_model_path}...")
            self.policy.load(init_model_path, self.device)

    def check(self) -> None:
        """Check the compatibility of selected modules."""
        self.logger.info("Invoking RLLTE Engine...")
        # sep line
        self.logger.info("=" * 80)
        self.logger.info(f"{'Tag'.ljust(NUMBER_OF_SPACES)} : {self.tag}")
        self.logger.info(f"{'Device'.ljust(NUMBER_OF_SPACES)} : {self.device_name}")
        self.logger.debug(f"{'Agent'.ljust(NUMBER_OF_SPACES)} : {self.__class__.__name__}")
        self.logger.debug(f"{'Encoder'.ljust(NUMBER_OF_SPACES)} : {self.encoder.__class__.__name__}")
        self.logger.debug(f"{'Policy'.ljust(NUMBER_OF_SPACES)} : {self.policy_name}")
        self.logger.debug(f"{'Storage'.ljust(NUMBER_OF_SPACES)} : {self.storage.__class__.__name__}")
        # class for `Distribution` and instance for `Noise`
        dist_name = self.dist.__name__ if isinstance(self.dist, type) else self.dist.__class__.__name__
        self.logger.debug(f"{'Distribution'.ljust(NUMBER_OF_SPACES)} : {dist_name}")

        # check augmentation and intrinsic reward
        if self.aug is not None:
            self.logger.debug(f"{'Augmentation'.ljust(NUMBER_OF_SPACES)} : True, {self.aug.__class__.__name__}")
        else:
            self.logger.debug(f"{'Augmentation'.ljust(NUMBER_OF_SPACES)} : False")

        if self.pretraining:
            assert self.irs is not None, "When the pre-training mode is turned on, an intrinsic reward must be specified!"

        if self.irs is not None:
            self.logger.debug(f"{'Intrinsic Reward'.ljust(NUMBER_OF_SPACES)} : True, {self.irs.__class__.__name__}")
        else:
            self.logger.debug(f"{'Intrinsic Reward'.ljust(NUMBER_OF_SPACES)} : False")

        if self.pretraining:
            self.logger.info(f"{'Pre-training Mode'.ljust(NUMBER_OF_SPACES)} : On")

        # sep line
        self.logger.debug("=" * 80)

    def set(
        self,
        encoder: Optional[Any] = None,
        policy: Optional[Any] = None,
        storage: Optional[Any] = None,
        distribution: Optional[Any] = None,
        augmentation: Optional[Any] = None,
        reward: Optional[Any] = None,
    ) -> None:
        """Set a module for the agent.

        Args:
            encoder (Optional[Any]): An encoder of `rllte.xploit.encoder` or a custom encoder.
            policy (Optional[Any]): A policy of `rllte.xploit.policy` or a custom policy.
            storage (Optional[Any]): A storage of `rllte.xploit.storage` or a custom storage.
            distribution (Optional[Any]): A distribution of `rllte.xplore.distribution` or a custom distribution.
            augmentation (Optional[Any]): An augmentation of `rllte.xplore.augmentation` or a custom augmentation.
            reward (Optional[Any]): A reward of `rllte.xplore.reward` or a custom reward.

        Returns:
            None.
        """
        if encoder is not None:
            assert isinstance(encoder, Encoder), "The `encoder` must be a subclass of `BaseEncoder`!"
            if self.encoder is not None:
                assert (
                    self.encoder.feature_dim == encoder.feature_dim
                ), "The feature dimension of `encoder` must be equal to the previous one!"
            self.encoder = encoder

        if policy is not None:
            assert isinstance(policy, Policy), "The `policy` must be a subclass of `BasePolicy`!"
            self.policy = policy

        if storage is not None:
            assert isinstance(storage, Storage), "The `storage` must be a subclass of `BaseStorage`!"
            self.storage = storage

        if distribution is not None:
            try:
                assert issubclass(distribution, Distribution), "The `distribution` must be a subclass of `BaseDistribution`!"
            except TypeError:
                assert isinstance(distribution, Distribution), "The `noise` must be a subclass of `BaseDistribution`!"
            self.dist = distribution

        if augmentation is not None:
            assert isinstance(augmentation, Augmentation), "The `augmentation` must be a subclass of `BaseAugmentation`!"
            self.aug = augmentation

        if reward is not None:
            assert isinstance(reward, IntrinsicRewardModule), "The `reward` must be a subclass of `BaseIntrinsicRewardModule`!"
            self.irs = reward

    def mode(self, training: bool = True) -> None:
        """Set the training mode.

        Args:
            training (bool): True (training) or False (evaluation).

        Returns:
            None.
        """
        self.training = training
        self.policy.train(training)

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """Update function of the agent."""

    @abstractmethod
    def train(
        self,
        num_train_steps: int,
        init_model_path: Optional[str],
        log_interval: int,
        eval_interval: int,
        num_eval_episodes: int,
        th_compile: bool,
    ) -> None:
        """Training function.

        Args:
            num_train_steps (int): The number of training steps.
            init_model_path (Optional[str]): The path of the initial model.
            log_interval (int): The interval of logging.
            eval_interval (int): The interval of evaluation.
            num_eval_episodes (int): The number of evaluation episodes.
            th_compile (bool): Whether to use `th.compile` or not.

        Returns:
            None.
        """

    @abstractmethod
    def eval(self, num_eval_episodes: int) -> Optional[Dict[str, float]]:
        """Evaluation function.

        Args:
            num_eval_episodes (int): The number of evaluation episodes.

        Returns:
            The evaluation results.
        """
