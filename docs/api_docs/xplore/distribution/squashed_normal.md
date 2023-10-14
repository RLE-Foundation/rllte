#


## SquashedNormal
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L67)
```python 

```


---
Squashed normal distribution for `Box` tasks.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L91)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L103)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L116)
```python
.mean()
```

---
Return the transformed mean.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L124)
```python
.mode()
```

---
Returns the mode of the distribution.

### .log_prob
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/squashed_normal.py/#L128)
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
