#


## OrnsteinUhlenbeck
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck.py/#L8)
```python 
OrnsteinUhlenbeck(
   mu: Tensor, sigma: Tensor, low: float = -1.0, high: float = 1.0, eps: float = 1e-06,
   theta: float = 0.15, dt: float = 0.01, initial_noise: Optional[Tensor] = None
)
```


---
Ornstein Uhlenbeck action noise.
Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


**Args**

* **mu**  : Mean of the distribution.
* **sigma**  : Standard deviation of the distribution.
* **low**  : Lower bound for action range.
* **high**  : Upper bound for action range.
* **eps**  : A constant for clamping.
* **theta**  : Rate of mean reversion.
* **dt**  : Timestep for the noise.
* **initial_noise**  : Initial value for the noise output, (if None: 0)


**Returns**

Ornstein-Uhlenbeck noise instance.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck.py/#L45)
```python
.reset()
```

---
Reset the Ornstein Uhlenbeck noise, to the initial position

### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/ornstein_uhlenbeck.py/#L53)
```python
.sample(
   clip: float = None, sample_shape = torch.Size()
)
```

---
Generates a sample_shape shaped sample


**Args**

* **clip**  : Range for noise truncation operation.
* **sample_shape**  : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.
