## Minimum Config
As shown in [Quick Start](quick_start.md), Hsuanwu uses [Hydra](https://hydra.cc/) to manage RL applications elegantly. To build a RL application, the 
minimum configuration file can be
``` yaml title="minimum_config.yaml"
device: cuda:0         # Device (cpu, cuda, ...).
seed: 1                # Random seed for reproduction.
num_train_steps: 5000  # Number of training steps.

agent:
  name: DrQv2          # The agent name.
```
This file will indicate the running device, random seed, training steps, and RL agent.

## Complete Parameter Support List
If you want to further specify other parameters, this is a complete list of parameters:
``` yaml title="complete_config.yaml"
####### Train setup
device: cuda:0                   # Device (cpu, cuda, ...) on which the code should be run.
seed: 1                          # Random seed for reproduction.
pretraining: false               # Turn on the pre-training mode.
init_model_path: ...             # Path of initial model parameters.
num_train_steps: 250000          # Number of training steps.
num_init_steps: 2000             # Number of initial exploration steps, only for `off-policy` agents.
####### Test setup
test_every_steps: 5000           # Testing interval, only for `off-policy` agents.
test_every_episodes: 10          # Testing interval, only for `on-policy` agents.
num_test_episodes: 10            # Number of testing episodes.
####### Choose the encoder
encoder:
 name: TassaCnnEncoder           # The encoder name. Supported types: https://docs.hsuanwu.dev/api/
 feature_dim: 50                 # The dimension of extracted features.
 ...
####### Choose the agent
learner:
 name: DrQv2                     # The agent name. Supported types: https://docs.hsuanwu.dev/api/
 ...
####### Choose the storage
storage:
 name: NStepReplayStorage        # The storage name. Supported types: https://docs.hsuanwu.dev/api/
 ...
####### Choose the distribution
distribution:
 name: TruncatedNormalNoise      # The distribution name. Supported types: https://docs.hsuanwu.dev/api/
 ...
####### Choose the augmentation
augmentation:
 name: RandomShift               # The augmentation name. Supported types: https://docs.hsuanwu.dev/api/
 ...
####### Choose the intrinsic reward
reward:
 name: RE3                       # The reward name. Supported types: https://docs.hsuanwu.dev/api/
 ...

hydra:
run:
  dir: ./logs/${experiment}/${now:%Y.%m.%d}/${now:%H%M%S}_${hydra.job.override_dirname} # Specify the output directory.
job:
  chdir: true # Change the working working.
```
For a specified module, you can further specify its internal parameters based on [API Documentation](https://docs.hsuanwu.dev/api/).
## Override Values
``` yaml title="example.yaml"
experiment: drqv2_dmc     # Experiment ID.
device: cuda:0            # Device (cpu, cuda, ...).
seed: 1                   # Random seed for reproduction.
num_train_steps: 5000     # Number of training steps.

agent:
  name: DrQv2             # The agent name.
```
For a written config, if you want to override some values, it suffices to:
``` sh
python train.py seed=7
```
Meanwhile, use `+` to add a new value if it isn't in the config:
``` sh
python train.py seed=7 +num_init_steps=3000
```

## Multirun
[Hydra](https://hydra.cc/) allows us to conveniently run the same application with multiple different configurations:
``` sh
python train.py --multirun seed=7,8,9

[2023-04-30 13:28:00,466][HYDRA] Launching 3 jobs locally
[2023-04-30 13:28:00,351][HYDRA] 	#0 : seed=7
[2023-04-30 13:28:01,679][HYDRA] 	#0 : seed=8
[2023-04-30 13:28:02,218][HYDRA] 	#0 : seed=9
```