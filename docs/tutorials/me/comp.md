# Performance Comparison of Multiple Algorithms

<div class="badge">
<a href="https://colab.research.google.com/github/RLE-Foundation/rllte/blob/main/examples/model_evaluation.ipynb">
<img src="../../../assets/images/colab-logo.svg" style="height: 32px; vertical-align:middle;">
Open in Colab
</a>
</div>

<div class="badge">
<a href="https://github.com/RLE-Foundation/rllte/blob/main/examples/model_evaluation.ipynb">
<img src="../../../assets/images/github-logo.svg" style="height: 32px; vertical-align:middle;">
View on GitHub
</a>
</div>

## Download Data
Suppose we want to evaluate algorithm performance on the [Procgen](https://github.com/openai/procgen) benchmark. First, download the data from 
[rllte-hub](https://hub.rllte.dev/):
``` py title="example.py"
# load packages
from rllte.evaluation import Performance, Comparison, min_max_normalize
from rllte.hub.datasets import Procgen, Atari
import numpy as np
# load scores
procgen = Procgen()
procgen_scores = procgen.load_scores()
print(procgen_scores.keys())
# get ppo-normalized scores
ppo_norm_scores = dict()
MIN_SCORES = np.zeros_like(procgen_scores['ppo'])
MAX_SCORES = np.mean(procgen_scores['ppo'], axis=0)
for algo in procgen_scores.keys():
    ppo_norm_scores[algo] = min_max_normalize(procgen_scores[algo],
                                              min_scores=MIN_SCORES,
                                              max_scores=MAX_SCORES)

# Output:
# dict_keys(['ppg', 'mixreg', 'ppo', 'idaac', 'plr', 'ucb-drac'])
```
For each algorithm, this will return a `NdArray` of size (`10` x `16`) where scores[n][m] represent the score on run `n` of task `m`.

## Performance Comparison
`Comparison` module allows you to compare the performance between two algorithms:
``` py title="example.py"
comp = Comparison(scores_x=ppo_norm_scores['PPG'],
                  scores_y=ppo_norm_scores['PPO'],
                  get_ci=True)
comp.compute_poi()

# Output:
# (0.8153125, array([[0.779375  ], [0.85000781]]))
```
This indicates the overall probability of imporvement of `PPG` over `PPO` is `0.8153125`.

Available metrics:

|Metric|Remark|
|:-|:-|
|`.compute_poi`|Compute the overall probability of imporvement of algorithm `X` over `Y`.|