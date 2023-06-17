# Benchmarks

**rllte-hub** provides a large number of reusable datasets and models of representative RL benchmarks. All the files 
are deposited on the [Hugging Face](https://huggingface.co) platform, view them by 

- [https://hub.rllte.dev/](https://hub.rllte.dev/) or
- [https://huggingface.co/RLE-Foundation](https://huggingface.co/RLE-Foundation).

## Datasets

`rllte.hub.datasets` provide test scores and learning cures of various RL algorithms on different benchmarks. 

### `.load_scores`
Suppose we want to evaluate algorithm performance on the [Procgen](https://github.com/openai/procgen) benchmark. Here is an example:

``` py title="example.py"
from rllte.hub.datasets import Procgen

procgen = Procgen()
procgen_scores = procgen.load_scores()
print(procgen_scores['PPO'].shape)

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

`rllte.hub.models` provide trained models of various RL algorithms on different benchmarks. Suppose we want to load an `PPO` agent trained on 
[Procgen](https://github.com/openai/procgen) benchmark, here is an example:

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

`rllte.hub.apps` provide fast-api for training RL agents with one-line command. Suppose we want to train an `PPO` agent on 
[Procgen](https://github.com/openai/procgen) benchmark, it suffices to run the following command:
``` sh
python -m rllte.hub.apps.ppo_procgen \
    --env_id bigfish \
    --seed 1 \
    --device "cuda"
```

## Support List
<table><thead><tr><th></th><th>PPO</th><th>DAAC</th><th>SAC</th><th>DDPG</th><th>IMPALA</th><th>DrAC</th><th>DrQ-v2</th><th>DrQ</th></tr></thead><tbody><tr><td><a href="https://www.jair.org/index.php/jair/article/download/10819/25823" target="_blank" rel="noopener noreferrer">Atari Games</a></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA" target="_blank" rel="noopener noreferrer">PyBullet</a></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://arxiv.org/pdf/1801.00690" target="_blank" rel="noopener noreferrer">DeepMind Control Suite</a></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="http://proceedings.mlr.press/v119/cobbe20a/cobbe20a.pdf" target="_blank" rel="noopener noreferrer">Procgen Games</a></td><td>💯 (25M)<br>📊 (25M)<br>🤖 (25M)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://github.com/Farama-Foundation/Minigrid" target="_blank" rel="noopener noreferrer">MiniGrid Games</a></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><a href="https://robosuite.ai/" target="_blank" rel="noopener noreferrer">Robosuite</a></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></tbody></table>

!!! tip
    - **🐌**: Incoming.
    - **(25M)**: 2.5e+6 training steps.
    - **💯Scores**: Available final scores.
    - **📊Curves**: Available training curves.
    - **🤖Models**: Available trained models.