#


## DiagonalGaussian
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L7)
```python 
DiagonalGaussian(
   loc: th.Tensor, scale: th.Tensor
)
```


---
Diagonal Gaussian distribution for 'Box' tasks.


**Args**

* **loc** (Tensor) : The mean of the distribution (often referred to as mu).
* **scale** (Tensor) : The standard deviation of the distribution (often referred to as sigma).


**Returns**

Squashed normal distribution instance.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L25)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L37)
```python
.rsample(
   sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of
reparameterized samples if the distribution parameters are batched.


**Args**

* **sample_shape** (Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L50)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L55)
```python
.mode()
```

---
Returns the mode of the distribution.

### .stddev
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L60)
```python
.stddev()
```

---
Returns the standard deviation of the distribution.

### .variance
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L65)
```python
.variance()
```

---
Returns the variance of the distribution.

### .log_prob
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L69)
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

### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L80)
```python
.reset()
```

---
Reset the distribution.

### .entropy
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/diagonal_gaussian.py/#L84)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.
