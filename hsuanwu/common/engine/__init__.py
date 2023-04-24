import gymnasium as gym
import omegaconf

from hsuanwu.xploit.agent import ALL_DEFAULT_CFGS, ALL_MATCH_KEYS

from .distributed_trainer import DistributedTrainer
from .off_policy_trainer import OffPolicyTrainer
from .on_policy_trainer import OnPolicyTrainer


class HsuanwuEngine:
    """Hsuanwu RL engine.

    Args:
        cfgs (DictConfig): Dict config for configuring RL algorithms.
        train_env (Env): A Gym-like environment for training.
        test_env (Env): A Gym-like environment for testing.

    Returns:
        Off-policy trainer instance.
    """

    def __init__(self, cfgs: omegaconf.DictConfig, train_env: gym.Env, test_env: gym.Env = None) -> None:
        assert hasattr(cfgs.agent, "name"), "The agent name must be specified!"

        if cfgs.agent.name not in ALL_DEFAULT_CFGS.keys():
            raise NotImplementedError(f"Unsupported agent {cfgs.agent.name}, see https://docs.hsuanwu.dev/overview/api/.")

        if ALL_MATCH_KEYS[cfgs.agent.name]["trainer"] == "OnPolicyTrainer":
            self.trainer = OnPolicyTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
        elif ALL_MATCH_KEYS[cfgs.agent.name]["trainer"] == "OffPolicyTrainer":
            self.trainer = OffPolicyTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
        elif ALL_MATCH_KEYS[cfgs.agent.name]["trainer"] == "DistributedTrainer":
            self.trainer = DistributedTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
        else:
            raise NotImplementedError(f"Unsupported trainer {cfgs.agent.name}, see https://docs.hsuanwu.dev/overview/api/.")

    def invoke(self):
        """Training function."""
        self.trainer.train()
