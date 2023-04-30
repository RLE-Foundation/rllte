## Defining the Environment
To use custom environments in Hsuanwu, it suffices to follow the [gymnasium](https://gymnasium.farama.org/) interface and prepare your environment following [Make your own custom environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#). Next, write a 
`make_env` function like
``` py title="make_env"
def make_env():
    def _thunk():
        return gym.make("Acrobot-v1")
    return _thunk
```


## Use `HsuanwuEnvWrapper`
In Hsuanwu, the environments are assumed to be ***vectorized*** and a `HsuanwuEnvWrapper` is
``` py title="example.py"
from hsuanwu.env.utils import HsuanwuEnvWrapper
# create vectorized environments
num_envs = 7
gym_env = gym.vector.SyncVectorEnv([make_env() for _ in range(num_envs)])
gym_env = gym.wrappers.RecordEpisodeStatistics(gym_env)
# wrap the environments
train_env = HsuanwuEnvWrapper(gym_env, device='cpu')
```
After that, you can use the custom environment in application directly.