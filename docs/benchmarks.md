---
hide:
  - navigation
---

# Benchmarks

**rllte-hub** provides a large number of reusable datasets and models of representative RL benchmarks. All the files 
are deposited on the [Hugging Face](https://huggingface.co) platform, view them by 

- [https://hub.rllte.dev/](https://hub.rllte.dev/) or
- [https://huggingface.co/RLE-Foundation](https://huggingface.co/RLE-Foundation).

| Module | Remark |
|:-|:-|
|`rllte.hub.datasets`|Provide **test scores** and **learning cures** of various RL algorithms on different benchmarks. |
|`rllte.hub.models`|Provide **trained models** of various RL algorithms on different benchmarks.|
|`rllte.hub.apps`|Provide **fast-APIs** for training RL agents with one-line command.|

## Support list



| Benchmark | Algorithm | Remark | Reproduction | Reference |
|:-|:-|:-|:-|:-|
|[Atari Games](https://www.jair.org/index.php/jair/article/download/10819/25823)|PPO|**50M**, 💯📊🤖| `.ppo_atari` | [Paper]() |
|[PyBullet](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)|PPO|**2M**, 💯📊🤖| `.ppo_bullet` | [Paper](https://proceedings.mlr.press/v164/raffin22a/raffin22a.pdf) |
||SAC|**1M**, 💯📊🤖| `.sac_bullet` | [Paper](https://proceedings.mlr.press/v164/raffin22a/raffin22a.pdf) |
|[DeepMind Control (Pixel)](https://arxiv.org/pdf/1801.00690)|DrQ-v2|**1M**, 💯📊🤖| `.drqv2_dmc_pixel` | [Paper](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com) |
|[DeepMind Control (State)](https://arxiv.org/pdf/1801.00690)|SAC|**10M**, 💯📊🤖| `.sac_dmc_state` | |
||DDPG|**10M**, 💯📊🤖| `.sac_dmc_state` | |
|[Procgen Games](http://proceedings.mlr.press/v119/cobbe20a/cobbe20a.pdf)|PPO|**25M**, 💯📊🤖| `ppo_procgen`| [Paper](http://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf) |
||PPO|**25M**, 💯📊🤖| `ppo_procgen_envpool`| [Paper](http://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf) |
|[MiniGrid Games](https://github.com/Farama-Foundation/Minigrid)||

!!! tip
    - **🐌**: Incoming.
    - **(25M)**: 25 million training steps.
    - **💯Scores**: Available final scores.
    - **📊Curves**: Available training curves.
    - **🤖Models**: Available trained models.

## Datasets

### `.load_scores`
Suppose we want to evaluate algorithm performance on the [Procgen](https://github.com/openai/procgen) benchmark. Here is an example:

``` py title="example.py"
from rllte.hub.datasets import Procgen

procgen = Procgen()
procgen_scores = procgen.load_scores()
print(procgen_scores['ppo'].shape)

# Output:
# (10, 16)
```
For each algorithm, this will return a `NdArray` of size (`10` x `16`) where `scores[n][m]` represent the score on run `n` of task `m`.

### `.load_curves`

Meanwhile, `.load_curves` will return the learning curves by a Python `Dict` like:

``` py
curves = {
    "ppo": {
        "train": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...}, 
        "eval": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...}, 
    },
    "daac": {
        "train": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...}, 
        "eval": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...}, 
    },
    ...
}
```
A code example for loading curves of the [Procgen](https://github.com/openai/procgen) benchmark:
``` py title="example.py"
from rllte.hub.datasets import Procgen

if __name__ == "__main__":
    # load data
    procgen = Procgen()
    curves = procgen.load_curves()

    print(curves['ppo']['train']['bigfish'].shape)
    print(curves['ppo']['eval']['bigfish'].shape)

# Output:
# (10, 1525)
# (10, 153)
```

## Models

Suppose we want to load an `PPO` agent trained on [Procgen](https://github.com/openai/procgen) benchmark, here is an example:

``` py title="example.py"
from rllte.hub.models import Procgen
from rllte.env import make_procgen_env
import torch as th
import numpy as np

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env_id = "starpilot"
    seed = 1
    # download the model
    procgen = Procgen()
    agent = procgen.load_models(agent="ppo",
                                env_id=env_id,
                                seed=seed,
                                device=device)
    # create env
    env = make_procgen_env(env_id=env_id, device=device, num_envs=1, seed=seed)
    # evaluate the model
    obs, infos = env.reset(seed=seed)
    # run the model
    episode_rewards, episode_steps = list(), list()
    while len(episode_rewards) < 10:
        # the exported model outputs logits of the action distribution
        action = th.softmax(agent(obs), dim=1).argmax(dim=1)
        obs, rewards, terminateds, truncateds, infos = env.step(action)

        if "episode" in infos:
            indices = np.nonzero(infos["episode"]["l"])
            episode_rewards.extend(infos["episode"]["r"][indices].tolist())
            episode_steps.extend(infos["episode"]["l"][indices].tolist())
    
    print(f"mean episode reward: {np.mean(episode_rewards)}")
    print(f"mean episode length: {np.mean(episode_steps)}")

# output:
mean episode reward: 30.0
mean episode length: 296.1
```

## Applications
Suppose we want to train an `PPO` agent on [Procgen](https://github.com/openai/procgen) benchmark, it suffices to run the following command:
``` sh
python -m rllte.hub.apps.ppo_procgen \
    --env_id bigfish \
    --seed 1 \
    --device "cuda"
```
All the results of `rllte.hub.datasets` and `rllte.hub.models` were trained via `rllte.hub.apps`, and all the hyper-parameters can be found in the reference.