## Write Configuration File
Hsuanwu supports the intrinsic reward-driven exploration and observation augmentation by default. To invoke them, it suffices to 
add the following contents in the config:
``` yaml title="config.yaml"
experiment: drqv2_dmc     # Experiment ID.
device: cuda:0            # Device (cpu, cuda, ...).
seed: 1                   # Random seed for reproduction.
num_train_steps: 5000     # Number of training steps.

agent:
  name: DrQv2             # The agent name.
```
The adjusted config:
``` yaml title="new_config.yaml"
experiment: drqv2_dmc     # Experiment ID.
device: cuda:0            # Device (cpu, cuda, ...).
seed: 1                   # Random seed for reproduction.
num_train_steps: 5000     # Number of training steps.

agent:
  name: DrQv2             # The agent name.

####### Choose the augmentation
augmentation:
 name: RandomShift        # The augmentation name. Supported types: https://docs.hsuanwu.dev/api/
 ...
####### Choose the intrinsic reward
reward:
 name: RE3                # The reward name. Supported types: https://docs.hsuanwu.dev/api/
 ...
```
For supported modules, see [API Documentation](https://docs.hsuanwu.dev/api/).

## Intro to Intrinsic Reward Modules
Due to the large differences in the calculation of different intrinsic reward methods, Hsuanwu has the following rules:

1. The environments are assumed to be ***vectorized***;
2. The ***compute_irs*** function of each intrinsic reward module has a mandatory argument ***samples***, which is a dict like:
     - obs (n_steps, n_envs, *obs_shape) <class 'torch.Tensor'>
     - actions (n_steps, n_envs, action_shape) <class 'torch.Tensor'>
     - rewards (n_steps, n_envs) <class 'torch.Tensor'>
     - next_obs (n_steps, n_envs, *obs_shape) <class 'torch.Tensor'>

Take RE3 for instance, it computes the intrinsic reward for each state based on the Euclidean distance between the state and 
its $k$-nearest neighbor within a mini-batch. Thus it suffices to provide ***obs*** data to compute the reward. The following code provides a usage example of RE3:
``` py title="example.py"
from hsuanwu.xplore.reward import RE3
from hsuanwu.env import make_dmc_env
import torch as th

if __name__ == '__main__':
    num_envs = 7
    num_steps = 128
    # create env
    env = make_dmc_env(env_id="cartpole_balance", num_envs=num_envs)
    print(env.observation_space, env.action_space)
    # create RE3 instance
    re3 = RE3(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    # compute intrinsic rewards
    obs = th.rand(size=(num_steps, num_envs, *env.observation_space.shape))
    intrinsic_rewards = re3.compute_irs(samples={'obs': obs})
    
    print(intrinsic_rewards.shape, type(intrinsic_rewards))
    print(intrinsic_rewards)

# Output:
# {'shape': [9, 84, 84]} {'shape': [1], 'type': 'Box', 'range': [-1.0, 1.0]}
# torch.Size([128, 7]) <class 'torch.Tensor'>
```