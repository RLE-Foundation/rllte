## Build Application

Hsuanwu uses [Hydra](hydra.cc) to manage RL applications elegantly. For example, 
we want to use [DrQ-v2](https://openreview.net/forum?id=_SJ-_yyes8) to solve a task of DeepMind Control Suite, and we only need the following two steps:

1. Write a `config.yaml` file in your working directory like:
``` yaml title="config.yaml"
experiment: drqv2_dmc     # Experiment ID.
device: cuda:0            # Device (cpu, cuda, ...).
seed: 1                   # Random seed for reproduction.
num_train_steps: 5000     # Number of training steps.

agent:
  name: DrQv2             # The agent name.
```

2. Write a train.py file like:
``` py title="train.py"
import hydra # Use Hydra to manage experiments
from hsuanwu.common.engine import HsuanwuEngine # Import Hsuanwu engine

train_env = make_dmc_env(env_id='cartpole_balance') # Create train env
test_env = make_dmc_env(env_id='cartpole_balance') # Create test env [Optional]

@hydra.main(version_base=None, config_path='./', config_name='config')
def main(cfgs):
    engine = HsuanwuEngine(cfgs=cfgs, train_env=train_env, test_env=test_env) # Initialize engine
    engine.invoke() # Start training

if __name__ == '__main__':
    main()
```

## Start Training

Run `train.py` and you will see the following output:
<div align=center>
<img src='../../assets/images/rl_training.png'>
</div>

!!! info "Read the logs"
    - **S**: Number of environment steps. Note that `S` isn't equal to the number of frames in visual tasks, and `number_of_frames=number_of_steps * number_of_action_repeats`
    - **E**: Number of environment episodes.
    - **L**: Average episode length.
    - **R**: Average episode reward.
    - **FPS**: Training FPS.
    - **T**: Time costs.

## Output/Working Directory
You can specify the working directory by  `config.yaml`, or an `outputs` folder will be used by default.
``` yaml title="config.yaml"
experiment: drqv2_dmc     # Experiment ID.
device: cuda:0            # Device (cpu, cuda, ...).
seed: 1                   # Random seed for reproduction.
num_train_steps: 5000     # Number of training steps.

agent:
  name: DrQv2             # The agent name.

hydra:
run:
  dir: ./logs/${experiment}/${now:%Y.%m.%d}/${now:%H%M%S}_${hydra.job.override_dirname} # Specify the output directory.
job:
  chdir: true # Change the working working.
```

## Load the Trained Model
Once the training is finished, you can find the trained model `agent.pth` in the subfolder `model` of the specified working directory.

``` py title="play.py"
import torch as th

# load the model and specify the map location
agent = th.load("agent.pth", map_location=th.device('cpu'))
obs = th.zeros(size=(1, 9, 84, 84))
action = agent(obs)
print(action)

# Output: tensor([[-1.0000]], grad_fn=<TanhBackward0>)
```