# Pre-training with Intrinsic Rewards

## Pre-training
Currently, **RLLTE** only supports online pre-training via intrinsic reward. To turn on the pre-training mode, 
it suffices to write a `train.py` like:
```py title="train.py"
from rllte.agent import PPO
from rllte.env import make_atari_env
from rllte.xplore.reward import RE3

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_atari_env(device=device)
    eval_env = make_atari_env(device=device)
    # create agent and turn on pre-training mode
    agent = PPO(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="ppo_atari",
                pretraining=True)
    # create intrinsic reward
    re3 = RE3(observation_space=env.observation_space,
              action_space=env.action_space,
              device=device)
    # set the reward module
    agent.set(reward=re3)
    # start training
    agent.train(num_train_steps=5000)
```
Run `train.py` and you'll see the pre-training mode is on:
``` sh
[08/04/2023 05:05:54 PM] - [INFO.] - Invoking RLLTE Engine...
[08/04/2023 05:05:54 PM] - [INFO.] - ================================================================================
[08/04/2023 05:05:54 PM] - [INFO.] - Tag               : ppo_atari
[08/04/2023 05:05:54 PM] - [INFO.] - Device            : NVIDIA GeForce RTX 3090
[08/04/2023 05:05:54 PM] - [DEBUG] - Agent             : PPO
[08/04/2023 05:05:54 PM] - [DEBUG] - Encoder           : MnihCnnEncoder
[08/04/2023 05:05:54 PM] - [DEBUG] - Policy            : OnPolicySharedActorCritic
[08/04/2023 05:05:54 PM] - [DEBUG] - Storage           : VanillaRolloutStorage
[08/04/2023 05:05:54 PM] - [DEBUG] - Distribution      : Categorical
[08/04/2023 05:05:54 PM] - [DEBUG] - Augmentation      : False
[08/04/2023 05:05:54 PM] - [DEBUG] - Intrinsic Reward  : True, RE3
[08/04/2023 05:05:54 PM] - [INFO.] - Pre-training Mode : On
[08/04/2023 05:05:54 PM] - [DEBUG] - ================================================================================
...
```

!!! tip
    When the pre-training mode is on, a `reward` module must be specified!
    
For all supported reward modules, see [API Documentation](https://docs.rllte.dev/api/).

## Fine-tuning
Once the pre-training is finished, you can find the model parameters in the `pretrained` subfolder of the working directory. To 
load the parameters, just turn off the pre-training mode and write the `train.py` like

```py title="train.py"
from rllte.agent import PPO
from rllte.env import make_atari_env

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_atari_env(device=device)
    eval_env = make_atari_env(device=device)
    # create agent and turn off pre-training mode
    agent = PPO(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="ppo_atari",
                pretraining=False)
    # start training
    agent.train(num_train_steps=5000,
                init_model_path="/export/yuanmingqi/code/rllte/logs/ppo_atari/2023-06-05-02-42-12/pretrained/pretrained.pth")
```
Run `train.py` and you'll see the pre-trained model parameters are loaded:
``` sh
[08/04/2023 05:07:52 PM] - [INFO.] - Invoking RLLTE Engine...
[08/04/2023 05:07:52 PM] - [INFO.] - ================================================================================
[08/04/2023 05:07:52 PM] - [INFO.] - Tag               : ppo_atari
[08/04/2023 05:07:52 PM] - [INFO.] - Device            : NVIDIA GeForce RTX 3090
[08/04/2023 05:07:53 PM] - [DEBUG] - Agent             : PPO
[08/04/2023 05:07:53 PM] - [DEBUG] - Encoder           : MnihCnnEncoder
[08/04/2023 05:07:53 PM] - [DEBUG] - Policy            : OnPolicySharedActorCritic
[08/04/2023 05:07:53 PM] - [DEBUG] - Storage           : VanillaRolloutStorage
[08/04/2023 05:07:53 PM] - [DEBUG] - Distribution      : Categorical
[08/04/2023 05:07:53 PM] - [DEBUG] - Augmentation      : False
[08/04/2023 05:07:53 PM] - [DEBUG] - Intrinsic Reward  : False
[08/04/2023 05:07:53 PM] - [DEBUG] - ================================================================================
[08/04/2023 05:07:53 PM] - [INFO.] - Loading Initial Parameters from ./logs/ppo_atari/...
...
```