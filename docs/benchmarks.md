# Benchmarks

## HsuanwuHub
Hsuanwu provides a large number of reusable data and models of representative RL benchmarks. Install `HsuanwuHub` by 
``` sh
pip install hsuanwuhub
```
Suppose we want to evaluate algorithm performance on the [Procgen](https://github.com/openai/procgen) benchmark.
``` py title="example.py"
from hsuanwuhub import datasets

procgen = datasets.Procgen()
procgen_scores = procgen.load_scores()
print(procgen_scores['PPO'].shape)

# Output:
# (10, 16)
```
For each algorithm, this will return a `NdArray` of size (`10` x `16`) where scores[n][m] represent the score on run `n` of task `m`.

## Fast Training
In `HsuanwuHub`, we preset a large number of RL applications to realize fast training. For example, 
we want to use [DrQ-v2](https://openreview.net/forum?id=_SJ-_yyes8) to solve a task of [DeepMind Control Suite](https://github.com/deepmind/dm_control). Then we can run the following command to perform training directly:
``` sh
python -m hsuanwuhub.train \
    task=drqv2_dmc_pixel \
    device=cuda:0 \
    num_train_steps=50000
```

## Support List
|Benchmark|Training Steps|Scores|Curves|Models|
|:-|:-|:-|:-|:-|
|[Atari Games](https://www.jair.org/index.php/jair/article/download/10819/25823)|100K|✔️|✔️|🐌|
|[PyBullet Robotics Environments](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)|🐌|🐌|🐌|🐌|
|[DeepMind Control Suite](https://arxiv.org/pdf/1801.00690)|500K|✔️|✔️|🐌|
|[Procgen Games](http://proceedings.mlr.press/v119/cobbe20a/cobbe20a.pdf)|25M|✔️|🐌|🐌|
|[MiniGrid Games](https://github.com/Farama-Foundation/Minigrid)|🐌|🐌|🐌|🐌|

!!! tip
    - **🐌**: Incoming.
    - **Scores**: Available final scores.
    - **Curves**: Available training curves.
    - **Visual**: Available trained models.

Find all the benchmark results in [https://hub.hsuanwu.dev/](https://hub.hsuanwu.dev/).