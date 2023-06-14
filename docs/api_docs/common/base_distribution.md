#


## BaseDistribution
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L31)
```python 

```


---
Abstract base class of distributions.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L39)
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

### .rsample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L51)
```python
.rsample(
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L63)
```python
.log_prob(
   value: th.Tensor
)
```

---
Returns the log of the probability density/mass function evaluated at `value`.


**Args**

* **value** (th.Tensor) : The value to be evaluated.


**Returns**

The log_prob value.

### .entropy
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L74)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L78)
```python
.reset()
```

---
Reset the distribution.

### .mean
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L82)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L86)
```python
.mode()
```

---
Returns the mode of the distribution.

### .stddev
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L90)
```python
.stddev()
```

---
Returns the standard deviation of the distribution.

### .variance
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_distribution.py/#L94)
```python
.variance()
```

---
Returns the variance of the distribution.
