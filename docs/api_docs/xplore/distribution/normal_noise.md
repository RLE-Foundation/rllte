#


## NormalNoise
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/normal_noise.py\#L8)
```python 
NormalNoise(
   mu: float = 0.0, sigma: float = 1.0, stddev_schedule: str = 'linear(1.0, 0.1,
   100000)', stddev_clip: float = 0.3
)
```


---
Gaussian action noise.


**Args**

* **mu** (float) : mean of the noise (often referred to as mu).
* **sigma** (float) : standard deviation of the noise (often referred to as sigma).
* **stddev_schedule** (str) : Use the exploration std schedule.


**Returns**

Gaussian action noise instance.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/normal_noise.py\#L33)
```python
.sample(
   clip: bool = False, sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.


**Args**

* **clip** (bool) : Whether to perform noise truncation.
* **sample_shape** (Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .rsample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/normal_noise.py\#L52)
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/normal_noise.py\#L64)
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/normal_noise.py\#L75)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/normal_noise.py\#L79)
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

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/normal_noise.py\#L95)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/normal_noise.py\#L100)
```python
.mode()
```

---
Returns the mode of the distribution.
