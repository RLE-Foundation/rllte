# Data Augmentation

**rllte** supports the intrinsic reward-driven exploration and observation augmentation by default, and users can invoke them in all the 
implemented algorithms.

## Using Observation Augmentation
**rllte** implements the augmentation modules via a PyTorch-NN manner, and both imaged-based and state-based observations are support. A code example is:
```py title="example.py"
from rllte.xploit.agent import PPO
from rllte.env import make_atari_env
from rllte.xplore.augmentation import RandomCrop

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_atari_env(device=device)
    eval_env = make_atari_env(device=device)
    # create agent
    agent = PPO(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="ppo_atari")
    # create augmentation module
    random_crop = RandomCrop()
    # set the module
    agent.set(augmentation=random_crop)
    # start training
    agent.train(num_train_steps=5000)
```
Run `example.py` and you'll see the augmentation module is invoked:
<div align=center>
<img src='../../assets/images/data_augmentation1.png'>
</div>

## Using Intrinsic Reward
Due to the large differences in the calculation of different intrinsic reward methods, rllte has the following rules:

1. The environments are assumed to be ***vectorized***;
2. The ***compute_irs*** function of each intrinsic reward module has a mandatory argument ***samples***, which is a dict like:
     - obs (n_steps, n_envs, *obs_shape) <class 'torch.Tensor'>
     - actions (n_steps, n_envs, *action_shape) <class 'torch.Tensor'>
     - rewards (n_steps, n_envs) <class 'torch.Tensor'>
     - next_obs (n_steps, n_envs, *obs_shape) <class 'torch.Tensor'>

Take RE3 for instance, it computes the intrinsic reward for each state based on the Euclidean distance between the state and 
its $k$-nearest neighbor within a mini-batch. Thus it suffices to provide ***obs*** data to compute the reward. The following code provides a usage example of RE3:
``` py title="example.py"
from rllte.xploit.agent import PPO
from rllte.env import make_atari_env
from rllte.xplore.reward import RE3

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_atari_env(device=device)
    eval_env = make_atari_env(device=device)
    # create agent
    agent = PPO(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="ppo_atari")
    # create intrinsic reward
    re3 = RE3(observation_space=env.observation_space,
              action_space=env.action_space,
              device=device)
    # set the module
    agent.set(reward=re3)
    # start training
    agent.train(num_train_steps=5000)
```
Run `example.py` and you'll see the intrinsic reward module is invoked:
<div align=center>
<img src='../../assets/images/data_augmentation2.png'>
</div>