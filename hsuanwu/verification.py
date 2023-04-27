import gymnasium as gym
from omegaconf import OmegaConf
from hsuanwu.common.engine import HsuanwuEngine
from hsuanwu.env.utils import HsuanwuEnvWrapper
from termcolor import colored

def make_env():
    def _thunk():
        return gym.make("Acrobot-v1")

    return _thunk

gym_env = gym.vector.SyncVectorEnv([make_env() for _ in range(1)])
gym_env = gym.wrappers.RecordEpisodeStatistics(gym_env)
train_env = HsuanwuEnvWrapper(gym_env, device='cpu')

cfgs = {
    'experiment': 'Verification',
    'device': 'cpu',
    'seed': 1,
    'num_train_steps': 5000,
    'agent': {
        'name': 'PPO'
    },
    'encoder': {
        'name': 'IdentityEncoder'
    },
}
cfgs = OmegaConf.create(cfgs)

if __name__ == '__main__':
    engine = HsuanwuEngine(
        cfgs=cfgs,
        train_env=train_env
    )
    try:
        engine.invoke()
        print(colored('Verification Passed!'.upper(), "green", attrs=["bold"]))
    except RuntimeError:
        print(colored('Verification failed!'.upper(), "red", attrs=["bold"]))