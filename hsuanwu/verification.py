import gymnasium as gym
from omegaconf import OmegaConf
from termcolor import colored

from hsuanwu.common.engine import HsuanwuEngine
from hsuanwu.env.utils import HsuanwuEnvWrapper


def make_env():
    def _thunk():
        return gym.make("Acrobot-v1")

    return _thunk


cfgs = {
    "experiment": "Verification",
    "device": "cpu",
    "seed": 1,
    "num_train_steps": 5000,
    "agent": {"name": "PPO"},
    "encoder": {"name": "IdentityEncoder"},
}
cfgs = OmegaConf.create(cfgs)

if __name__ == "__main__":
    train_env = HsuanwuEnvWrapper(make_env, num_envs=1, device="cpu")
    engine = HsuanwuEngine(cfgs=cfgs, train_env=train_env)
    try:
        engine.invoke()
        print(colored("Verification Passed!".upper(), "green", attrs=["bold"]))
    except RuntimeError:
        print(colored("Verification failed!".upper(), "red", attrs=["bold"]))
