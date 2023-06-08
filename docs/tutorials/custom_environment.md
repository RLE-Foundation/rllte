# Custom Environment

## Defining the Environment
To use custom environments in **rllte**, it suffices to follow the [gymnasium](https://gymnasium.farama.org/) interface and prepare your environment following [Tutorials: Make Your Own Custom Environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#). A example is:
``` py title="example.py"
import gymnasium as gym
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, total_length) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            shape=(9, 84, 84),
            high=255.0,
            low=0.,
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            shape=(7,),
            high=1.,
            low=-1.,
            dtype=np.float32
        )
        self.total_length = total_length
        self.count = 0

    def step(self, action):
        obs = self.observation_space.sample()
        reward = np.random.rand()
        if self.count < self.total_length:
            terminated = truncated = False
        else:
            terminated = truncated = True
        info = {"discount": 0.99}
        self.count += 1

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.count = 0
        return self.observation_space.sample(), {"discount": 0.99}
```

## Use `make_rllte_env`
In **rllte**, the environments are assumed to be ***vectorized*** and a `make_rllte_env` function is used to warp the environments:
``` py title="example.py"
from rllte.env.utils import make_rllte_env
# create vectorized environments
env = make_rllte_env(env_id=CustomEnv, 
                     device=device, 
                     env_kwargs={'total_length': 499} # set env arguments
                     )
```
After that, you can use the custom environment in application directly.
``` py title="train.py"
from rllte.xploit.agent import DrQv2
from rllte.env.utils import make_rllte_env

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_rllte_env(env_id=CustomEnv, 
                        device=device, 
                        env_kwargs={'total_length': 499} # set env arguments
                        )
    eval_env = make_rllte_env(env_id=CustomEnv, 
                            device=device, 
                            env_kwargs={'total_length': 499} # set env arguments
                            )
    agent = DrQv2(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="drqv2_dmc_pixel")
    agent.train(num_train_steps=5000)
```