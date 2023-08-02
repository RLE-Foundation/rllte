#


## OrnsteinUhlenbeckNoise
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L34)
```python 
OrnsteinUhlenbeckNoise(
   loc: float = 0.0, scale: float = 1.0, low: float = -1.0, high: float = 1.0,
   eps: float = 1e-06, theta: float = 0.15, dt: float = 0.01,
   stddev_schedule: str = 'linear(1.0, 0.1, 100000)', stddev_clip: float = 0.3
)
```


---
Ornstein Uhlenbeck action noise.
Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


**Args**

* **loc** (float) : mean of the noise (often referred to as mu).
* **scale** (float) : standard deviation of the noise (often referred to as sigma).
* **low** (float) : The lower bound of the noise.
* **high** (float) : The upper bound of the noise.
* **eps** (float) : A small value to avoid numerical instability.
* **theta** (float) : The rate of mean reversion.
* **dt** (float) : Timestep for the noise.
* **stddev_schedule** (str) : Use the exploration std schedule.
* **stddev_clip** (float) : The exploration std clip range.


**Returns**

Ornstein-Uhlenbeck noise instance.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L80)
```python
.reset(
   noiseless_action: th.Tensor, step: int = 0
)
```

---
Reset the noise instance.


**Args**

* **noiseless_action** (th.Tensor) : Unprocessed actions.
* **step** (int) : Global training step that can be None when there is no noise schedule.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L103)
```python
.sample(
   clip: bool = False, sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped sample


**Args**

* **clip** (bool) : Range for noise truncation operation.
* **sample_shape** (th.Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .mean
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L138)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L143)
```python
.mode()
```

---
Returns the mode of the distribution.
