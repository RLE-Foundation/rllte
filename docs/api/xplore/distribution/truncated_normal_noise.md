#


## TruncatedNormalNoise
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/truncated_normal_noise.py\#L9)
```python 
TruncatedNormalNoise(
   mu: float = 0.0, sigma: float = 1.0, stddev_schedule: str = 'linear(1.0, 0.1,
   100000)', stddev_clip: float = 0.3
)
```


---
Truncated normal action noise. See Section 3.1 of "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning".


**Args**

* **mu** (float) : mean of the noise (often referred to as mu).
* **sigma** (float) : standard deviation of the noise (often referred to as sigma).
* **stddev_schedule** (str) : Use the exploration std schedule.
* **stddev_clip** (float) : The exploration std clip range.


**Returns**

Truncated normal noise instance.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/truncated_normal_noise.py\#L38)
```python
.sample(
   clip: bool = False, sample_shape: TorchSize = torch.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.


**Args**

* **clip** (bool) : Whether to perform noise truncation.
* **sample_shape** (TorchSize) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .rsample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/truncated_normal_noise.py\#L61)
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/truncated_normal_noise.py\#L73)
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/truncated_normal_noise.py\#L84)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/truncated_normal_noise.py\#L88)
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

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/truncated_normal_noise.py\#L106)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/truncated_normal_noise.py\#L111)
```python
.mode()
```

---
Returns the mode of the distribution.
