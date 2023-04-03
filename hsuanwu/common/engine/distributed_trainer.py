from collections import deque

import hydra
import numpy as np
import torch

from hsuanwu.common.engine import BasePolicyTrainer, utils
from hsuanwu.common.logger import *
from hsuanwu.common.typing import DictConfig, Env


from hsuanwu.xploit.learner import IMPALALearner
from hsuanwu.xploit.storage import DistributedStorage

class DistributedTrainer(BasePolicyTrainer):
    """Trainer for on-policy algorithms.

    Args:
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.
        cfgs (DictConfig): Dict config for configuring RL algorithms.

    Returns:
        On-policy trainer instance.
    """
    def __init__(self, train_env: Env, test_env: Env, cfgs: DictConfig) -> None:
        super().__init__(train_env, test_env, cfgs)

        # create shared storages
        self._shared_storages = DistributedStorage(
            device=self._device
        )

