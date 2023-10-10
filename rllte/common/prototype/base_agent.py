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
from typing import Dict, Optional, Any

import numpy as np
import pynvml
import torch as th

# auxiliary modules
from rllte.common.logger import Logger
from rllte.common.preprocessing import process_action_space, process_observation_space
from rllte.common.prototype.base_augmentation import BaseAugmentation as Augmentation
from rllte.common.prototype.base_distribution import BaseDistribution as Distribution
from rllte.common.prototype.base_encoder import BaseEncoder as Encoder
from rllte.common.prototype.base_policy import BasePolicy as Policy
from rllte.common.prototype.base_reward import BaseIntrinsicRewardModule as IntrinsicRewardModule
from rllte.common.prototype.base_storage import BaseStorage as Storage
from rllte.common.timer import Timer
from rllte.common.type_alias import VecEnv
from rllte.common.utils import get_npu_name

NUMBER_OF_SPACES = 17

# try to load torch_npu
try:
    import torch_npu as torch_npu  # type: ignore

    if th.npu.is_available():  # type: ignore
        NPU_AVAILABLE = True
except Exception:
    NPU_AVAILABLE = False


class BaseAgent(ABC):
    """Base class of the agent.

    Args:
        env (VecEnv): Vectorized environments for training.
        eval_env (VecEnv): Vectorized environments for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.

    Returns:
        Base agent instance.
    """

    def __init__(
        self,
        env: VecEnv,
        eval_env: Optional[VecEnv] = None,
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
        self.encoder: Optional[Encoder] = None
        self.policy: Optional[Policy] = None
        self.storage: Optional[Storage] = None
        self.dist: Optional[Distribution] = None
        self.aug: Optional[Augmentation] = None
        self.irs: Optional[IntrinsicRewardModule] = None
        self.attr_names = ("encoder", "policy", "storage", "dist", "aug", "irs")
        self.module_names = {key: None for key in self.attr_names}

        # training tracking
        self.pretraining = pretraining
        self.global_step = 0
        self.global_episode = 0
        self.metrics: Dict[str, Any] = {}

    def freeze(self, **kwargs) -> None:
        """Freeze the agent and get ready for training."""
        assert self.policy is not None, "A policy must be specified!"
        # freeze the structure of the agent
        self.policy.freeze(encoder=self.encoder, dist=self.dist)
        # torch compilation
        if kwargs.get("th_compile", False):
            self.policy = th.compile(self.policy)  # type: ignore
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
        # check essential modules
        for attr_name in ["encoder", "policy", "storage", "dist"]:
            assert getattr(self, attr_name) is not None, f"The `{attr_name}` must be specified!"
        # print basic info
        self.logger.info("Invoking RLLTE Engine...")
        ## sep line
        self.logger.info("=" * 80)
        self.logger.info(f"{'Tag'.ljust(NUMBER_OF_SPACES)} : {self.tag}")
        self.logger.info(f"{'Device'.ljust(NUMBER_OF_SPACES)} : {self.device_name}")
        # output module info
        titles = ("Agent", "Encoder", "Policy", "Storage", "Distribution", "Augmentation", "Intrinsic Reward")
        for i in range(len(titles)):
            if titles[i] == "Agent":
                name = self.__class__.__name__
            else:
                name = self.module_names[self.attr_names[i - 1]]  # type: ignore[assignment]
            self.logger.debug(f"{titles[i].ljust(NUMBER_OF_SPACES)} : {name}")
        # check pre-training setting
        if self.pretraining:
            assert self.irs is not None, "When the pre-training mode is turned on, an intrinsic reward must be specified!"
            self.logger.info(f"{'Pre-training Mode'.ljust(NUMBER_OF_SPACES)} : On")
        # sep line
        self.logger.debug("=" * 80)

    def set(
        self,
        encoder: Optional[Encoder] = None,
        policy: Optional[Policy] = None,
        storage: Optional[Storage] = None,
        distribution: Optional[Distribution] = None,
        augmentation: Optional[Augmentation] = None,
        reward: Optional[IntrinsicRewardModule] = None,
    ) -> None:
        """Set a module for the agent.

        Args:
            encoder (Optional[Encoder]): An encoder of `rllte.xploit.encoder` or a custom encoder.
            policy (Optional[Policy]): A policy of `rllte.xploit.policy` or a custom policy.
            storage (Optional[Storage]): A storage of `rllte.xploit.storage` or a custom storage.
            distribution (Optional[Distribution]): A distribution of `rllte.xplore.distribution`
                or a custom distribution.
            augmentation (Optional[Augmentation]): An augmentation of `rllte.xplore.augmentation`
                or a custom augmentation.
            reward (Optional[IntrinsicRewardModule]): A reward of `rllte.xplore.reward` or a custom reward.

        Returns:
            None.
        """
        args = [encoder, policy, storage, distribution, augmentation, reward]
        arg_names = ("encoder", "policy", "storage", "distribution", "augmentation", "reward")
        types = (Encoder, Policy, Storage, Distribution, Augmentation, IntrinsicRewardModule)

        for i in range(len(args)):
            if args[i] is not None:
                assert isinstance(args[i], types[i]), f"The `{arg_names[i]}` must be a subclass of `{types[i].__name__}`!"
                setattr(self, self.attr_names[i], args[i])
                self.module_names[self.attr_names[i]] = args[i].__class__.__name__  # type: ignore[assignment]

        # overwrite the encoder
        if encoder is not None and self.encoder is not None:
            assert (
                self.encoder.feature_dim == encoder.feature_dim
            ), "The feature dimension of `encoder` must be equal to the previous one!"

    def mode(self, training: bool = True) -> None:
        """Set the training mode.

        Args:
            training (bool): True (training) or False (evaluation).

        Returns:
            None.
        """
        self.training = training
        self.policy.train(training)  # type: ignore

    def save(self) -> None:
        """Save the agent."""
        if self.pretraining:
            save_dir = Path.cwd() / "pretrained"
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = Path.cwd() / "model"
            save_dir.mkdir(exist_ok=True)

        self.policy.save(path=save_dir, pretraining=self.pretraining, global_step=self.global_step)  # type: ignore

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update function of the agent."""

    @abstractmethod
    def train(
        self,
        num_train_steps: int,
        init_model_path: Optional[str],
        log_interval: int,
        eval_interval: int,
        save_interval: int,
        num_eval_episodes: int,
        th_compile: bool,
    ) -> None:
        """Training function.

        Args:
            num_train_steps (int): The number of training steps.
            init_model_path (Optional[str]): The path of the initial model.
            log_interval (int): The interval of logging.
            eval_interval (int): The interval of evaluation.
            save_interval (int): The interval of saving model.
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
