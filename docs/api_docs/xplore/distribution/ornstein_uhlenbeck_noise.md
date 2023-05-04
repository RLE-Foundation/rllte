#


## OrnsteinUhlenbeckNoise
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L9)
```python 
OrnsteinUhlenbeckNoise(
   loc: float = 0.0, scale: float = 1.0, theta: float = 0.15, dt: float = 0.01,
   stddev_schedule: str = 'linear(1.0, 0.1, 100000)'
)
```


---
Ornstein Uhlenbeck action noise.
Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


**Args**

* **loc** (float) : mean of the noise (often referred to as mu).
* **scale** (float) : standard deviation of the noise (often referred to as sigma).
* **theta** (float) : Rate of mean reversion.
* **dt** (float) : Timestep for the noise.


**Returns**

Ornstein-Uhlenbeck noise instance.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L42)
```python
.reset(
   noiseless_action: th.Tensor, step: int = 0
)
```

---
Reset the noise instance.


**Args**

* **noiseless_action** (Tensor) : Unprocessed actions.
* **step** (int) : Global training step that can be None when there is no noise schedule.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L59)
```python
.sample(
   clip: bool = False, sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped sample


**Args**

* **clip** (bool) : Range for noise truncation operation.
* **sample_shape** (Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L90)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L95)
```python
.mode()
```

---
Returns the mode of the distribution.

### .rsample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L99)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L111)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L122)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .stddev
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L127)
```python
.stddev()
```

---
Returns the standard deviation of the distribution.

### .variance
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py/#L132)
```python
.variance()
```

---
Returns the variance of the distribution.
