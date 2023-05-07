#


## Categorical
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L7)
```python 
Categorical(
   logits: th.Tensor
)
```


---
Categorical distribution for sampling actions for 'Discrete' tasks.

**Args**

* **logits** (Tensor) : The event log probabilities (unnormalized).


**Returns**

Categorical distribution instance.


**Methods:**


### .probs
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L24)
```python
.probs()
```

---
Return probabilities.

### .logits
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L29)
```python
.logits()
```

---
Returns the unnormalized log probabilities.

### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L33)
```python
.sample(
   sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.


**Args**

* **sample_shape** (TorchSize) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .log_prob
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L45)
```python
.log_prob(
   actions: th.Tensor
)
```

---
Returns the log of the probability density/mass function evaluated at actions.


**Args**

* **actions** (Tensor) : The actions to be evaluated.


**Returns**

The log_prob value.

### .entropy
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L56)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .mode
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L61)
```python
.mode()
```

---
Returns the mode of the distribution.

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L66)
```python
.mean()
```

---
Returns the mean of the distribution.

### .stddev
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L71)
```python
.stddev()
```

---
Returns the standard deviation of the distribution.

### .variance
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L76)
```python
.variance()
```

---
Returns the variance of the distribution.

### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L80)
```python
.reset()
```

---
Reset the distribution.

### .rsample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L84)
```python
.rsample(
   sample_shape: th.Size = ...
)
```

