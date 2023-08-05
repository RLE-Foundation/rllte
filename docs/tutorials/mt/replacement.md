# Module Replacement for An Implemented Algorithm

**RLLTE** allows developers to replace settled modules of implemented algorithms to make performance comparison and algorithm improvement.

## Use built-in modules
For instance, we want to use [PPO](https://arxiv.org/pdf/1707.06347) agent to solve [Atari](https://www.jair.org/index.php/jair/article/download/10819/25823) games, it suffices to write `train.py` like:
``` py title="train.py"
from rllte.agent import PPO
from rllte.env import make_atari_env

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
    # start training
    agent.train(num_train_steps=5000)
```
Run `train.py` and you'll see the following output:
``` sh
[08/04/2023 03:45:54 PM] - [INFO.] - Invoking RLLTE Engine...
[08/04/2023 03:45:54 PM] - [INFO.] - ================================================================================
[08/04/2023 03:45:54 PM] - [INFO.] - Tag               : ppo_atari
[08/04/2023 03:45:54 PM] - [INFO.] - Device            : NVIDIA GeForce RTX 3090
[08/04/2023 03:45:55 PM] - [DEBUG] - Agent             : PPO
[08/04/2023 03:45:55 PM] - [DEBUG] - Encoder           : MnihCnnEncoder
[08/04/2023 03:45:55 PM] - [DEBUG] - Policy            : OnPolicySharedActorCritic
[08/04/2023 03:45:55 PM] - [DEBUG] - Storage           : VanillaRolloutStorage
[08/04/2023 03:45:55 PM] - [DEBUG] - Distribution      : Categorical
[08/04/2023 03:45:55 PM] - [DEBUG] - Augmentation      : False
[08/04/2023 03:45:55 PM] - [DEBUG] - Intrinsic Reward  : False
[08/04/2023 03:45:55 PM] - [DEBUG] - ================================================================================
[08/04/2023 03:45:56 PM] - [EVAL.] - S: 0           | E: 0           | L: 23          | R: 24.000      | T: 0:00:02    
[08/04/2023 03:45:57 PM] - [TRAIN] - S: 1024        | E: 8           | L: 44          | R: 99.000      | FPS: 346.187   | T: 0:00:02    
[08/04/2023 03:45:58 PM] - [TRAIN] - S: 2048        | E: 16          | L: 58          | R: 207.000     | FPS: 514.168   | T: 0:00:03    
[08/04/2023 03:45:59 PM] - [TRAIN] - S: 3072        | E: 24          | L: 43          | R: 70.000      | FPS: 619.411   | T: 0:00:04    
[08/04/2023 03:46:00 PM] - [TRAIN] - S: 4096        | E: 32          | L: 43          | R: 67.000      | FPS: 695.523   | T: 0:00:05    
[08/04/2023 03:46:00 PM] - [INFO.] - Training Accomplished!
[08/04/2023 03:46:00 PM] - [INFO.] - Model saved at: /export/yuanmingqi/code/rllte/logs/ppo_atari/2023-08-04-03-45-54/model
```

Suppose we want to use a `ResNet-based` encoder, it suffices to replace the encoder module using `.set` function:
``` py title="train.py"
from rllte.agent import PPO
from rllte.env import make_atari_env
from rllte.xploit.encoder import EspeholtResidualEncoder

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_atari_env(device=device)
    eval_env = make_atari_env(device=device)
    # create agent
    feature_dim = 512
    agent = PPO(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="ppo_atari",
                feature_dim=feature_dim)
    # create a new encoder
    encoder = EspeholtResidualEncoder(
        observation_space=env.observation_space,
        feature_dim=feature_dim)
    # set the new encoder
    agent.set(encoder=encoder)
    # start training
    agent.train(num_train_steps=5000)
```
Run `train.py` and you'll see the old `MnihCnnEncoder` has been replaced by `EspeholtResidualEncoder`:
``` sh
[08/04/2023 03:46:38 PM] - [INFO.] - Invoking RLLTE Engine...
[08/04/2023 03:46:38 PM] - [INFO.] - ================================================================================
[08/04/2023 03:46:38 PM] - [INFO.] - Tag               : ppo_atari
[08/04/2023 03:46:38 PM] - [INFO.] - Device            : NVIDIA GeForce RTX 3090
[08/04/2023 03:46:38 PM] - [DEBUG] - Agent             : PPO
[08/04/2023 03:46:38 PM] - [DEBUG] - Encoder           : EspeholtResidualEncoder
[08/04/2023 03:46:38 PM] - [DEBUG] - Policy            : OnPolicySharedActorCritic
[08/04/2023 03:46:38 PM] - [DEBUG] - Storage           : VanillaRolloutStorage
[08/04/2023 03:46:38 PM] - [DEBUG] - Distribution      : Categorical
[08/04/2023 03:46:38 PM] - [DEBUG] - Augmentation      : False
[08/04/2023 03:46:38 PM] - [DEBUG] - Intrinsic Reward  : False
[08/04/2023 03:46:38 PM] - [DEBUG] - ================================================================================
...
```
For more replaceable modules, please refer to [https://docs.rllte.dev/api/](https://docs.rllte.dev/api/).

## Using custom modules

Developers can also perform replacement using custom modules, see [Make A Custom Module](../custom/module.md) for more details.