# Model Evaluation
**rllte** provides evaluation methods based on:

> [Agarwal R, Schwarzer M, Castro P S, et al. Deep reinforcement learning at the edge of the statistical precipice[J]. Advances in neural information processing systems, 2021, 34: 29304-29320.](https://proceedings.neurips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)

We reconstruct and improve the code of the official repository [rliable](https://github.com/google-research/rliable), achieving higher convenience and efficiency.

## Download Data
- Suppose we want to evaluate algorithm performance on the [Procgen](https://github.com/openai/procgen) benchmark. First, download the data from 
[rllte-hub](https://hub.rllte.dev/):
``` py title="example.py"
# load packages
from rllte.evaluation import Performance, Comparison, min_max_normalize
from rllte.evaluation import *
from rllte.hub.datasets import Procgen, Atari
import numpy as np
# load scores
procgen = Procgen()
procgen_scores = procgen.load_scores()
print(procgen_scores.keys())
# PPO-Normalized scores
ppo_norm_scores = dict()
MIN_SCORES = np.zeros_like(procgen_scores['PPO'])
MAX_SCORES = np.mean(procgen_scores['PPO'], axis=0)
for algo in procgen_scores.keys():
    ppo_norm_scores[algo] = min_max_normalize(procgen_scores[algo],
                                              min_scores=MIN_SCORES,
                                              max_scores=MAX_SCORES)

# Output:
# dict_keys(['PPG', 'MixReg', 'PPO', 'IDAAC', 'PLR', 'UCB-DrAC'])
```
For each algorithm, this will return a `NdArray` of size (`10` x `16`) where scores[n][m] represent the score on run `n` of task `m`.

## Performance Evaluation
Import the performance evaluator:
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
|`.create_performance_profile`|Computes the performance profilies.|

## Performance Comparison
`Comparison` module allows you to compare the performance between two algorithms:
``` py title="example.py"
comp = Comparison(scores_x=ppo_norm_scores['PPG'],
                  scores_y=ppo_norm_scores['PPO'],
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
### `.plot_interval_estimates`
`.plot_interval_estimates` can plot verious metrics of algorithms with stratified confidence intervals. Take [Procgen](https://github.com/openai/procgen) for example, we want to plot four reliable metrics computed by `Performance` evaluator:
```py title="example.py"
aggregate_performance_dict = {
    "MEAN": {},
    "MEDIAN": {},
    "IQM": {},
    "OG": {}
}
for algo in ppo_norm_scores.keys():
    perf = Performance(scores=ppo_norm_scores[algo], get_ci=True)
    aggregate_performance_dict['MEAN'][algo] = perf.aggregate_mean()
    aggregate_performance_dict['MEDIAN'][algo] = perf.aggregate_median()
    aggregate_performance_dict['IQM'][algo] = perf.aggregate_iqm()
    aggregate_performance_dict['OG'][algo] = perf.aggregate_og()

fig, axes = plot_interval_estimates(aggregate_performance_dict,
                                    metric_names=['MEAN', 'MEDIAN', 'IQM', 'OG'],
                                    algorithms=['PPO', 'MixReg', 'UCB-DrAC', 'PLR', 'PPG', 'IDAAC'],
                                    xlabel="PPO-Normalized Score")
fig.savefig('./plot_interval_estimates1.png', format='png', bbox_inches='tight')

fig, axes = plot_interval_estimates(aggregate_performance_dict,
                        metric_names=['MEAN', 'MEDIAN'],
                        algorithms=['PPO', 'MixReg', 'UCB-DrAC', 'PLR', 'PPG', 'IDAAC'],
                        xlabel="PPO-Normalized Score")
fig.savefig('./plot_interval_estimates2.png', format='png', bbox_inches='tight')

fig, axes = plot_interval_estimates(aggregate_performance_dict,
                        metric_names=['MEAN', 'MEDIAN'],
                        algorithms=['PPO', 'MixReg', 'UCB-DrAC', 'PLR', 'PPG', 'IDAAC'],
                        xlabel="PPO-Normalized Score")
fig.savefig('./plot_interval_estimates2.png', format='png', bbox_inches='tight')
```
The output figures are:
<div align=center>
<img src='../../assets/images/plot_interval_estimates1.png' style="filter: drop-shadow(0px 0px 7px #000);">
<img src='../../assets/images/plot_interval_estimates2.png' style="filter: drop-shadow(0px 0px 7px #000);">
<img src='../../assets/images/plot_interval_estimates3.png' style="filter: drop-shadow(0px 0px 7px #000);">
</div>


### `.plot_probability_improvement`
`.plot_probability_improvement` plots probability of improvement with stratified confidence intervals. An example is:
```py title="example.py"
pairs = [['IDAAC', 'PPG'], ['IDAAC', 'UCB-DrAC'], ['IDAAC', 'PPO'],
    ['PPG', 'PPO'], ['UCB-DrAC', 'PLR'], 
    ['PLR', 'MixReg'], ['UCB-DrAC', 'MixReg'],  ['MixReg', 'PPO']]

probability_of_improvement_dict = {}
for pair in pairs:
    comp = Comparison(scores_x=ppo_norm_scores[pair[0]], 
                      scores_y=ppo_norm_scores[pair[1]],
                      get_ci=True)
    probability_of_improvement_dict['_'.join(pair)] = comp.compute_poi()

fig, ax = plot_probability_improvement(poi_dict=probability_of_improvement_dict)
fig.savefig('./plot_probability_improvement.png', format='png', bbox_inches='tight')
```
The output figure is:
<div align=center>
<img src='../../assets/images/plot_probability_improvement.png' style="filter: drop-shadow(0px 0px 7px #000);">
</div>

### `.plot_performance_profile`
`.plot_performance_profile` plots performance profiles with stratified confidence intervals. An example is:
```py title="example.py"
profile_dict = dict()
procgen_tau = np.linspace(0.5, 3.6, 101)

for algo in ppo_norm_scores.keys():
    perf = Performance(scores=ppo_norm_scores[algo], get_ci=True, reps=2000)
    profile_dict[algo] = perf.create_performance_profile(tau_list=procgen_tau)

fig, axes = plot_performance_profile(profile_dict, 
                         procgen_tau,
                         figsize=(7, 5),
                         xlabel=r'PPO-Normalized Score $(\tau)$',
                         )
fig.savefig('./plot_performance_profile.png', format='png', bbox_inches='tight')
```
The output figure is:
<div align=center>
<img src='../../assets/images/plot_performance_profile.png' style="filter: drop-shadow(0px 0px 7px #000);">
</div>

### `.plot_sample_efficiency_curve`
`.plot_sample_efficiency_curve` plots an aggregate metric with CIs as a function of environment frames. An example is:
```py title="example.py"
ale_all_frames_scores_dict = Atari().load_curves()
frames = np.array([1, 10, 25, 50, 75, 100, 125, 150, 175, 200]) - 1

sampling_dict = dict()
for algo in ale_all_frames_scores_dict.keys():
    sampling_dict[algo] = [[], [], []]
    for frame in frames:
        perf = Performance(ale_all_frames_scores_dict[algo][:, :, frame],
                           get_ci=True, 
                           reps=2000)
        value, CIs = perf.aggregate_iqm()
        sampling_dict[algo][0].append(value)
        sampling_dict[algo][1].append(CIs[0]) # lower bound
        sampling_dict[algo][2].append(CIs[1]) # upper bound
    
    sampling_dict[algo][0] = np.array(sampling_dict[algo][0]).reshape(-1)
    sampling_dict[algo][1] = np.array(sampling_dict[algo][1]).reshape(-1)
    sampling_dict[algo][2] = np.array(sampling_dict[algo][2]).reshape(-1)

algorithms = ['C51', 'DQN (Adam)', 'DQN (Nature)', 'Rainbow', 'IQN', 'REM', 'M-IQN', 'DreamerV2']
fig, axes = plot_sample_efficiency_curve(
    sampling_dict,
    frames+1, 
    figsize=(7, 4.5),
    algorithms=algorithms,
    xlabel=r'Number of Frames (in millions)',
    ylabel='IQM Human Normalized Score')
fig.savefig('./plot_sample_efficiency_curve.png', format='png', bbox_inches='tight')
```
The output figure is:
<div align=center>
<img src='../../assets/images/plot_sample_efficiency_curve.png' style="filter: drop-shadow(0px 0px 7px #000);">
</div>