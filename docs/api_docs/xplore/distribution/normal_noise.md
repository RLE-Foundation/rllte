#


## NormalNoise
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/normal_noise.py/#L36)
```python 
NormalNoise(
   mu: Union[float, th.Tensor] = 0.0, sigma: Union[float, th.Tensor] = 1.0,
   low: float = -1.0, high: float = 1.0, eps: float = 1e-06
)
```


---
Gaussian action noise.


**Args**

* **mu** (Union[float, th.Tensor]) : Mean of the noise.
* **sigma** (Union[float, th.Tensor]) : Standard deviation of the noise.
* **low** (float) : The lower bound of the noise.
* **high** (float) : The upper bound of the noise.
* **eps** (float) : A small value to avoid numerical instability.


**Returns**

Gaussian action noise instance.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/normal_noise.py/#L85)
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

### .mean
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/normal_noise.py/#L108)
```python
.mean()
```

---
Returns the mean of the distribution.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/distribution/normal_noise.py/#L113)
```python
.mode()
```

---
Returns the mode of the distribution.
