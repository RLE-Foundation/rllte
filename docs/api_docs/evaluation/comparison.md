#


## Comparison
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/evaluation/comparison.py\#L7)
```python 
Comparison(
   scores_x: np.ndarray, scores_y: np.ndarray
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


**Returns**

Comparer instance.


**Methods:**


### .compute_poi
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/evaluation/comparison.py\#L33)
```python
.compute_poi()
```

---
Compute the overall Probability of imporvement of algorithm `X` over `Y`.
