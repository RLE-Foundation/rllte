#


## Performance
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/performance.py/#L10)
```python 
Performance(
   scores: np.ndarray, get_ci: bool = False, method: str = 'percentile',
   task_bootstrap: bool = False, reps: int = 50000,
   confidence_interval_size: float = 0.95,
   random_state: Optional[random.RandomState] = None
)
```


---
Evaluate the performance of an algorithm. Based on:
https://github.com/google-research/rliable/blob/master/rliable/metrics.py


**Args**

* **scores** (NdArray) : A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
    represent the score on run `n` of task `m`.
* **get_ci** (bool) : Compute CIs or not.
* **method** (str) :  One of `basic`, `percentile`, `bc` (identical to `debiased`,
    `bias-corrected`), or `bca`.
* **task_bootstrap** (bool) :  Whether to perform bootstrapping over tasks in addition to
    runs. Defaults to False. See `StratifiedBoostrap` for more details.
* **reps** (int) : Number of bootstrap replications.
* **confidence_interval_size** (float) : Coverage of confidence interval.
* **random_state** (int) : If specified, ensures reproducibility in uncertainty estimates.


**Returns**

Performance evaluator.


**Methods:**


### .aggregate_mean
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/performance.py/#L48)
```python
.aggregate_mean()
```

---
Computes mean of sample mean scores per task.

### .aggregate_median
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/performance.py/#L61)
```python
.aggregate_median()
```

---
Computes median of sample mean scores per task.

### .aggregate_og
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/performance.py/#L74)
```python
.aggregate_og(
   gamma: float = 1.0
)
```

---
Computes optimality gap across all runs and tasks.


**Args**

* **gamma** (float) : Threshold for optimality gap. All scores above `gamma` are clipped
to `gamma`.


**Returns**

Optimality gap at threshold `gamma`.

### .aggregate_iqm
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/performance.py/#L94)
```python
.aggregate_iqm()
```

---
Computes the interquartile mean across runs and tasks.

### .get_interval_estimates
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/performance.py/#L106)
```python
.get_interval_estimates(
   scores: np.ndarray, metric: Callable
)
```

---
Computes interval estimation of the above performance evaluators.


**Args**

* **scores** (NdArray) : A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
    represent the score on run `n` of task `m`.
* **metric** (Callable) : One of the above performance evaluators used for estimation.


**Returns**

Confidence intervals.

### .create_performance_profile
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/performance.py/#L127)
```python
.create_performance_profile(
   tau_list: Union[List[float], np.ndarray], use_score_distribution: bool = True
)
```

---
Method for calculating performance profilies.


**Args**

* **tau_list** (Union[List[float], np.ndarray]) : List of 1D numpy array of threshold
    values on which the profile is evaluated.
* **use_score_distribution** (bool) : Whether to report score distributions or average
    score distributions.


**Returns**

Point and interval estimates of profiles evaluated at all thresholds in 'tau_list'.
