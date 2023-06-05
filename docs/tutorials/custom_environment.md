# Custom Environment

## Defining the Environment
To use custom environments in **rllte**, it suffices to follow the [gymnasium](https://gymnasium.farama.org/) interface and prepare your environment following [Make your own custom environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#). Next, write a 
`make_env` function like
``` py title="make_env"
def make_env():
    def _thunk():
        return gym.make("Acrobot-v1")
    return _thunk
```

## Use `VecEnvWrapper`
In **rllte**, the environments are assumed to be ***vectorized*** and a `VecEnvWrapper` is used to warp the environments:
``` py title="example.py"
from rllte.env.utils import VecEnvWrapper
# create vectorized environments
train_env = VecEnvWrapper(env_fn=make_env, num_envs=7, device='cpu')
```
After that, you can use the custom environment in application directly.
``` py title="train.py"
from rllte.xploit.agent import PPO
from rllte.env.utils import VecEnvWrapper
import gymnasium as gym

def make_env():
    def _thunk():
        return gym.make("Acrobot-v1")
    return _thunk

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = VecEnvWrapper(env_fn=make_env, num_envs=7, device=device)
    eval_env = VecEnvWrapper(env_fn=make_env, num_envs=7, device=device)
    # create agent
    agent = PPO(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="ppo_atari")
    # start training
    agent.train(num_train_steps=5000)
```