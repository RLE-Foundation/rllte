import gymnasium as gym
import omegaconf

from hsuanwu.xploit.agent import ALL_DEFAULT_CFGS

from .base_policy_trainer import BasePolicyTrainer as BasePolicyTrainer
from .distributed_trainer import DistributedTrainer as DistributedTrainer
from .off_policy_trainer import OffPolicyTrainer as OffPolicyTrainer
from .on_policy_trainer import OnPolicyTrainer as OnPolicyTrainer


class HsuanwuEngine:
    """Hsuanwu RL engine.

    Args:
        cfgs (DictConfig): Dict config for configuring RL algorithms.
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.

    Returns:
        Hsuanwu engine instance.
    """

    def __init__(self, cfgs: omegaconf.DictConfig, train_env: gym.Env, test_env: gym.Env = None) -> None:
        assert hasattr(cfgs.agent, "name"), "The agent name must be specified!"

        if cfgs.agent.name not in ALL_DEFAULT_CFGS.keys():
            raise NotImplementedError(f"Unsupported agent {cfgs.agent.name}, see https://docs.hsuanwu.dev/overview/api/.")

        if ALL_DEFAULT_CFGS[cfgs.agent.name]["MATCH_KEYS"]["trainer"] == "OnPolicyTrainer":
            self.trainer = OnPolicyTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
        elif ALL_DEFAULT_CFGS[cfgs.agent.name]["MATCH_KEYS"]["trainer"] == "OffPolicyTrainer":
            self.trainer = OffPolicyTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
        elif ALL_DEFAULT_CFGS[cfgs.agent.name]["MATCH_KEYS"]["trainer"] == "DistributedTrainer":
            self.trainer = DistributedTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
        else:
            raise NotImplementedError(f"Unsupported trainer {cfgs.agent.name}, see https://docs.hsuanwu.dev/overview/api/.")

    def invoke(self):
        """Invoke the engine to perform training."""
        self.trainer.train()
