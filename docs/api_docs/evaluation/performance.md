#


## Performance
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/evaluation/performance.py\#L7)
```python 
Performance(
   scores: np.ndarray
)
```


---
Evaluate the performance of an algorithm. Based on:
https://github.com/google-research/rliable/blob/master/rliable/metrics.py


**Args**

* **scores** (NdArray) : A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
    represent the score on run `n` of task `m`.


**Returns**

Performance evaluator.


**Methods:**


### .agg_mean
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/evaluation/performance.py\#L27)
```python
.agg_mean()
```

---
Computes mean of sample mean scores per task.

### .agg_median
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/evaluation/performance.py\#L32)
```python
.agg_median()
```

---
Computes median of sample mean scores per task.

### .agg_og
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/evaluation/performance.py\#L37)
```python
.agg_og(
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

### .agg_iqm
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/evaluation/performance.py\#L49)
```python
.agg_iqm()
```

---
Computes the interquartile mean across runs and tasks.

### .describe
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/evaluation/performance.py\#L53)
```python
.describe()
```

---
Compute all the evaluation metrics.
