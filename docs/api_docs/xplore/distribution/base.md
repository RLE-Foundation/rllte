#


## BaseDistribution
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L6)
```python 

```


---
Abstract base class of distributions.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L14)
```python
.sample(
   sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.


**Args**

* **sample_shape** (Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .rsample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L26)
```python
.rsample(
   sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.


**Args**

* **sample_shape** (Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .log_prob
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L38)
```python
.log_prob(
   value: th.Tensor
)
```

---
Returns the log of the probability density/mass function evaluated at `value`.


**Args**

* **value** (Tensor) : The value to be evaluated.


**Returns**

The log_prob value.

### .entropy
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L49)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L53)
```python
.reset()
```

---
Reset the distribution.

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L57)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L61)
```python
.mode()
```

---
Returns the mode of the distribution.

### .stddev
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L65)
```python
.stddev()
```

---
Returns the standard deviation of the distribution.

### .variance
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/base.py/#L69)
```python
.variance()
```

---
Returns the variance of the distribution.
