# Performance Evaluation of Single Algorithm

<div class="badge">
<a href="https://colab.research.google.com/github/RLE-Foundation/rllte/blob/main/examples/model_evaluation.ipynb">
<img src="../../../assets/images/colab-logo.svg" style="height: 32px; vertical-align:middle;">
Open in Colab
</a>
</div>

<div class="badge">
<a href="https://github.com/RLE-Foundation/rllte/blob/main/examples/model_evaluation.ipynb">
<img src="../../../assets/images/github-logo.svg" style="height: 32px; vertical-align:middle;">
&nbsp;&nbsp;View on GitHub
</a>
</div>

**RLLTE** provides evaluation methods based on:

> [Agarwal R, Schwarzer M, Castro P S, et al. Deep reinforcement learning at the edge of the statistical precipice[J]. Advances in neural information processing systems, 2021, 34: 29304-29320.](https://proceedings.neurips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)

We reconstruct and improve the code of the official repository [rliable](https://github.com/google-research/rliable), achieving higher convenience and efficiency.

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

## Performance Evaluation
Initialize the performance evaluator:
``` py title="example.py"
perf = Performance(scores=ppo_norm_scores['PPO'], 
                   get_ci=True # get confidence intervals
                   )
perf.aggregate_mean()

# Output:
# Computing confidence interval for aggregate MEAN...
# (1.0, array([[0.9737281 ], [1.02564405]]))
```
Available metrics:

|Metric|Remark|
|:-|:-|
|`.aggregate_mean`|Computes mean of sample mean scores per task.|
|`.aggregate_median`|Computes median of sample mean scores per task.|
|`.aggregate_og`|Computes optimality gap across all runs and tasks.|
|`.aggregate_iqm`|Computes the interquartile mean across runs and tasks.|
|`.create_performance_profile`|Computes the performance profiles.|