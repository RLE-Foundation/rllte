**rllte** provides evaluation methods based on:

> [Agarwal R, Schwarzer M, Castro P S, et al. Deep reinforcement learning at the edge of the statistical precipice[J]. Advances in neural information processing systems, 2021, 34: 29304-29320.](https://proceedings.neurips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)

We reconstruct and improve the code of the official repository [rliable](https://github.com/google-research/rliable), achieving higher convenience and efficiency.

## Download Data
- Suppose we want to evaluate algorithm performance on the [Procgen](https://github.com/openai/procgen) benchmark. First, download the data from 
[rllte-benchmark](https://hub.rllte.dev/):
``` sh
pip install rllte-benchmark
```
- Load data:
``` py title="example.py"
from rllte import datasets
from rllte.evaluation import Performance, Comparison, min_max_normalize
import numpy as np

procgen = datasets.Procgen()
procgen_scores = procgen.load_scores()
ppo_scores, ppg_scores = procgen_scores['PPO'], procgen_scores['PPG']
# PPO-Normalized scores
norm_ppo_scores = min_max_normalize(ppo_scores, 
                                    min_scores=np.zeros_like(ppo_scores), 
                                    max_scores=np.mean(ppo_scores, axis=0))
norm_ppg_scores = min_max_normalize(ppg_scores, 
                                    min_scores=np.zeros_like(ppo_scores), 
                                    max_scores=np.mean(ppo_scores, axis=0))
```
For each algorithm, this will return a `NdArray` of size (`10` x `16`) where scores[n][m] represent the score on run `n` of task `m`.

## Performance Evaluation
Import performance evaluator:
``` py title="example.py"
perf = Performance(scores=norm_ppo_scores, 
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

## Performance Comparison
`Comparison` module allows you to compare the performance between two algorithms:
``` py title="example.py"
comp = Comparison(scores_x=norm_ppg_scores,
                  scores_y=norm_ppo_scores,
                  get_ci=True)
comp.compute_poi()

# Output:
# Computing confidence interval for PoI...
# (0.8153125, array([[0.779375  ], [0.85000781]]))
```
This indicates the overall probability of imporvement of `PPG` over `PPO` is `0.8153125`.

Available metrics:

|Metric|Remark|
|:-|:-|
|`.compute_poi`|Compute the overall probability of imporvement of algorithm `X` over `Y`.|

## Visualization