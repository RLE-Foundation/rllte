#


## Comparison
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/evaluation/comparison.py/#L9)
```python 
Comparison(
   scores_x: np.ndarray, scores_y: np.ndarray, get_ci: bool = False,
   method: str = 'percentile', reps: int = 2000, confidence_interval_size: float = 0.95,
   random_state: Optional[random.RandomState] = None
)
```


---
Compare the performance between algorithms. Based on:
https://github.com/google-research/rliable/blob/master/rliable/metrics.py


**Args**

* **scores_x** (NdArray) : A matrix of size (`num_runs_x` x `num_tasks`) where scores[n][m]
    represent the score on run `n` of task `m` for algorithm `X`.
* **scores_y** (NdArray) : A matrix of size (`num_runs_y` x `num_tasks`) where scores[n][m]
    represent the score on run `n` of task `m` for algorithm `Y`.
* **get_ci** (bool) : Compute CIs or not.
* **method** (str) :  One of `basic`, `percentile`, `bc` (identical to `debiased`,
    `bias-corrected`), or `bca`.
* **reps** (int) : Number of bootstrap replications.
* **confidence_interval_size** (float) : Coverage of confidence interval.
* **random_state** (int) : If specified, ensures reproducibility in uncertainty estimates.


**Returns**

Comparer instance.


**Methods:**


### .compute_poi
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/evaluation/comparison.py/#L48)
```python
.compute_poi()
```

---
Compute the overall probability of imporvement of algorithm `X` over `Y`.

### .get_interval_estimates
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/evaluation/comparison.py/#L77)
```python
.get_interval_estimates(
   scores_x: np.array, scores_y: np.array, metric: Callable
)
```

---
Computes interval estimation of the above performance evaluators.


**Args**

* **scores_x** (NdArray) : A matrix of size (`num_runs_x` x `num_tasks`) where scores[n][m]
    represent the score on run `n` of task `m` for algorithm `X`.
* **scores_y** (NdArray) : A matrix of size (`num_runs_y` x `num_tasks`) where scores[n][m]
    represent the score on run `n` of task `m` for algorithm `Y`.
* **metric** (Callable) : One of the above performance evaluators used for estimation.


**Returns**

Confidence intervals.
