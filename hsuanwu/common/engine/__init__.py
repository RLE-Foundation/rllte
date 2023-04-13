from .base_policy_trainer import BasePolicyTrainer
from .distributed_trainer import DistributedTrainer
from .off_policy_trainer import OffPolicyTrainer
from .on_policy_trainer import OnPolicyTrainer

from hsuanwu.xploit.learner import ALL_DEFAULT_CFGS
from hsuanwu.common.typing import DictConfig, Env

_LEARNER_TO_TRAINER = {
    'OffPolicyTrainer': ['DrQv2Learner', 'SACLearner'],
    'OnPolicyTrainer': ['PPOLearner', 'PPGLearner'],
    'DistributedTrainer': ['IMPALALearner']
}

class Engine:
    """Hsuanwu RL engine.

    Args:
        cfgs (DictConfig): Dict config for configuring RL algorithms.
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.

    Returns:
        Off-policy trainer instance.
    """

    def __init__(self, cfgs: DictConfig, train_env: Env, test_env: Env = None) -> None:
        try:
            cfgs.learner.name
        except:
            raise ValueError(f"The learner name must be specified!")
        
        if cfgs.learner.name not in ALL_DEFAULT_CFGS.keys():
            raise NotImplementedError(
                f"Unsupported learner {cfgs.learner.name}, see https://docs.hsuanwu.dev/overview/api_overview/."
            )
        
        if cfgs.learner.name in _LEARNER_TO_TRAINER['OnPolicyTrainer']:
            self.trainer = OnPolicyTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
        elif cfgs.learner.name in _LEARNER_TO_TRAINER['OffPolicyTrainer']:
            self.trainer = OffPolicyTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
        elif cfgs.learner.name in _LEARNER_TO_TRAINER['DistributedTrainer']:
            self.trainer = DistributedTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
        else:
            raise NotImplementedError(f"Unsupported trainer {cfgs.learner.name}, see https://docs.hsuanwu.dev/overview/api_overview/.")
        
    def invoke(self):
        """Training function.
        """
        self.trainer.train()
    
