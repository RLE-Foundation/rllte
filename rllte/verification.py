import gymnasium as gym
from termcolor import colored

from rllte.env.utils import make_rllte_env
from rllte.xploit.agent import PPO


def make_env():
    def _thunk():
        return gym.make("Acrobot-v1")

    return _thunk


if __name__ == "__main__":
    env = make_rllte_env(env_id="Acrobot-v1", num_envs=1, device="cpu")
    agent = PPO(env=env, device="cpu", tag="verification")
    try:
        agent.train(num_train_steps=1000)
        print(colored("Verification Passed!".upper(), "green", attrs=["bold"]))
    except RuntimeError:
        print(colored("Verification failed!".upper(), "red", attrs=["bold"]))
