#


## MultiCategorical
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/multi_categorical.py/#L32)
```python 
MultiCategorical(
   logits: Tuple[th.Tensor, ...]
)
```


---
Categorical distribution for sampling actions for 'MultiDiscrete' tasks.


**Args**

* **logits** (Tuple[th.Tensor, ...]) : The event log probabilities (unnormalized).


**Returns**

Categorical distribution instance.


**Methods:**


### .probs
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/multi_categorical.py/#L47)
```python
.probs()
```

---
Return probabilities.

### .logits
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/multi_categorical.py/#L52)
```python
.logits()
```

---
Returns the unnormalized log probabilities.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/multi_categorical.py/#L56)
```python
.sample(
   sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.


**Args**

* **sample_shape** (th.Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .log_prob
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/multi_categorical.py/#L68)
```python
.log_prob(
   actions: th.Tensor
)
```

---
Returns the log of the probability density/mass function evaluated at actions.


**Args**

* **actions** (th.Tensor) : The actions to be evaluated.


**Returns**

The log_prob value.

### .entropy
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/multi_categorical.py/#L81)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/multi_categorical.py/#L86)
```python
.mode()
```

---
Returns the mode of the distribution.

### .mean
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/multi_categorical.py/#L91)
```python
.mean()
```

---
Returns the mean of the distribution.
