# Intrinsic Reward Shaping for Enhancing Exploration

Since **RLLTE** decouples RL algorithms into minimum primitives from the perspective of exploitation and exploration, intrinsic reward shaping is supported by default. Due to the large differences in the calculation of different intrinsic reward methods, rllte has the following rules:

1. The environments are assumed to be ***vectorized***;
2. The ***compute_irs*** function of each intrinsic reward module has a mandatory argument ***samples***, which is a dict like:
     - obs (n_steps, n_envs, *obs_shape) <class 'torch.Tensor'>
     - actions (n_steps, n_envs, *action_shape) <class 'torch.Tensor'>
     - rewards (n_steps, n_envs) <class 'torch.Tensor'>
     - next_obs (n_steps, n_envs, *obs_shape) <class 'torch.Tensor'>

Take RE3 for instance, it computes the intrinsic reward for each state based on the Euclidean distance between the state and 
its $k$-nearest neighbor within a mini-batch. Thus it suffices to provide ***obs*** data to compute the reward. The following code provides a usage example of RE3:
``` py title="example.py"
from rllte.agent import PPO
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
``` sh
[08/04/2023 03:54:10 PM] - [INFO.] - Invoking RLLTE Engine...
[08/04/2023 03:54:10 PM] - [INFO.] - ================================================================================
[08/04/2023 03:54:10 PM] - [INFO.] - Tag               : ppo_atari
[08/04/2023 03:54:10 PM] - [INFO.] - Device            : NVIDIA GeForce RTX 3090
[08/04/2023 03:54:11 PM] - [DEBUG] - Agent             : PPO
[08/04/2023 03:54:11 PM] - [DEBUG] - Encoder           : MnihCnnEncoder
[08/04/2023 03:54:11 PM] - [DEBUG] - Policy            : OnPolicySharedActorCritic
[08/04/2023 03:54:11 PM] - [DEBUG] - Storage           : VanillaRolloutStorage
[08/04/2023 03:54:11 PM] - [DEBUG] - Distribution      : Categorical
[08/04/2023 03:54:11 PM] - [DEBUG] - Augmentation      : False
[08/04/2023 03:54:11 PM] - [DEBUG] - Intrinsic Reward  : True, RE3
[08/04/2023 03:54:11 PM] - [DEBUG] - ================================================================================
```