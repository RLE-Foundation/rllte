#


## SquashedNormal
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L64)
```python 
SquashedNormal(
   loc: th.Tensor, scale: th.Tensor
)
```


---
Squashed normal distribution for Soft Actor-Critic learner.


**Args**

* **loc** (th.Tensor) : The mean of the distribution (often referred to as mu).
* **scale** (th.Tensor) : The standard deviation of the distribution (often referred to as sigma).


**Returns**

Squashed normal distribution instance.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L85)
```python
.sample(
   sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped
batch of samples if the distribution parameters are batched.


**Args**

* **sample_shape** (th.Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .rsample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L97)
```python
.rsample(
   sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped reparameterized sample or sample_shape shaped
batch of reparameterized samples if the distribution parameters are batched.


**Args**

* **sample_shape** (th.Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .mean
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L110)
```python
.mean()
```

---
Return the transformed mean.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L118)
```python
.mode()
```

---
Returns the mode of the distribution.

### .log_prob
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L122)
```python
.log_prob(
   actions: th.Tensor
)
```

---
Scores the sample by inverting the transform(s) and computing the score using
the score of the base distribution and the log abs det jacobian.

**Args**

* **actions** (th.Tensor) : The actions to be evaluated.


**Returns**

The log_prob value.

### .entropy
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L133)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .stddev
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L138)
```python
.stddev()
```

---
Returns the standard deviation of the distribution.

### .variance
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L143)
```python
.variance()
```

---
Returns the variance of the distribution.

### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L147)
```python
.reset()
```

---
Reset the distribution.
