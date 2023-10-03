#


## OrnsteinUhlenbeckNoise
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L36)
```python 
OrnsteinUhlenbeckNoise(
   mu: Union[float, th.Tensor] = 0.0, sigma: Union[float, th.Tensor] = 1.0,
   low: float = -1.0, high: float = 1.0, eps: float = 1e-06, theta: float = 0.15,
   dt: float = 0.01
)
```


---
Ornstein Uhlenbeck action noise.
Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


**Args**

* **mu** (Union[float, th.Tensor]) : Mean of the noise.
* **sigma** (Union[float, th.Tensor]) : Standard deviation of the noise.
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


### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L96)
```python
.sample(
   clip: Optional[float] = None, sample_shape: th.Size = th.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.


**Args**

* **clip** (Optional[float]) : The clip range of the sampled noises.
* **sample_shape** (th.Size) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L131)
```python
.reset()
```

---
Reset the noise.

### .mean
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L136)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/ornstein_uhlenbeck_noise.py/#L141)
```python
.mode()
```

---
Returns the mode of the distribution.
