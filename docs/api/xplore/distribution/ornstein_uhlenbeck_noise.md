#


## OrnsteinUhlenbeckNoise
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py\#L10)
```python 
OrnsteinUhlenbeckNoise(
   mu: float = 0.0, sigma: float = 1.0, theta: float = 0.15, dt: float = 0.01,
   stddev_schedule: str = 'linear(1.0, 0.1, 100000)'
)
```


---
Ornstein Uhlenbeck action noise.
Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


**Args**

* **mu** (float) : mean of the noise (often referred to as mu).
* **sigma** (float) : standard deviation of the noise (often referred to as sigma).
* **theta** (float) : Rate of mean reversion.
* **dt** (float) : Timestep for the noise.


**Returns**

Ornstein-Uhlenbeck noise instance.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py\#L43)
```python
.reset(
   noiseless_action: Tensor, step: int = None
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py\#L60)
```python
.sample(
   clip: float = None, sample_shape: TorchSize = torch.Size()
)
```

---
Generates a sample_shape shaped sample


**Args**

* **clip**  : Range for noise truncation operation.
* **sample_shape**  : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py\#L93)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py\#L98)
```python
.mode()
```

---
Returns the mode of the distribution.

### .rsample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py\#L102)
```python
.rsample(
   sample_shape: TorchSize = torch.Size()
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py\#L114)
```python
.log_prob(
   value: Tensor
)
```

---
Returns the log of the probability density/mass function evaluated at `value`.


**Args**

* **value** (Tensor) : The value to be evaluated.


**Returns**

The log_prob value.

### .entropy
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/ornstein_uhlenbeck_noise.py\#L125)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.
