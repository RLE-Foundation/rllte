# Quick Start

## On NVIDIA GPU
In **rllte**, users can build RL applications with very simple code. For example, 
we want to use [DrQ-v2](https://openreview.net/forum?id=_SJ-_yyes8) to solve a task of DeepMind Control Suite, and 
it suffices to write a `train.py` like:

``` py title="train.py"
# import `env` and `agent` api
from rllte.env import make_dmc_env 
from rllte.xploit.agent import DrQv2

if __name__ == "__main__":
    device = "cuda:0"
    # create env, and `eval_env` is optional
    env = make_dmc_env(env_id="cartpole_balance", device=device)
    eval_env = make_dmc_env(env_id="cartpole_balance", device=device)
    # create agent
    agent = DrQv2(env=env, 
                  eval_env=eval_env, 
                  device='cuda',
                  tag="drqv2_dmc_pixel")
    # start training
    agent.train(num_train_steps=5000)
```

Run `train.py` and you will see the following output:
<div align=center>
<img src='../../assets/images/rl_training_gpu.png' style="filter: drop-shadow(0px 0px 7px #000);">
</div>

!!! info "Read the logs"
    - **S**: Number of environment steps. Note that `S` isn't equal to the number of frames in visual tasks, and `number_of_frames=number_of_steps * number_of_action_repeats`
    - **E**: Number of environment episodes.
    - **L**: Average episode length.
    - **R**: Average episode reward.
    - **FPS**: Training FPS.
    - **T**: Time costs.

## On HUAWEI NPU
Similarly, if we want to train an agent on HUAWEI NPU, it suffices to replace `cuda` with `npu`:
``` py title="train.py"
# import `env` and `agent` api
from rllte.env import make_dmc_env 
from rllte.xploit.agent import DrQv2

if __name__ == "__main__":
    device = "npu:0"
    # create env, and `eval_env` is optional
    env = make_dmc_env(env_id="cartpole_balance", device=device)
    eval_env = make_dmc_env(env_id="cartpole_balance", device=device)
    # create agent
    agent = DrQv2(env=env, 
                  eval_env=eval_env, 
                  device='cuda',
                  tag="drqv2_dmc_pixel")
    # start training
    agent.train(num_train_steps=5000)
```

!!! info "Compatibility of NPU"
    Please refer to [https://docs.rllte.dev/api/](https://docs.rllte.dev/api/) for the compatibility of NPU.

Run `train.py` and you will see the following output:
<div align=center>
<img src='../../assets/images/rl_training_npu.png' style="filter: drop-shadow(0px 0px 7px #000);">
</div>

## Load the Trained Model
Once the training is finished, you can find `agent.pth` in the subfolder `model` of the specified working directory.

``` py title="play.py"
import torch as th

# load the model and specify the map location
agent = th.load("agent.pth", map_location=th.device('cpu'))
obs = th.zeros(size=(1, 9, 84, 84))
action = agent(obs)
print(action)

# Output: tensor([[-1.0000]], grad_fn=<TanhBackward0>)
```