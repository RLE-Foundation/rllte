#


## GaussianNoise
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/gaussian_noise.py/#L7)
```python 
GaussianNoise(
   mu: float = 0, sigma: float = 1.0
)
```


---
Gaussian noise operation for processing state-based observations.


**Args**

* **mu** (float or Tensor) : mean of the distribution.
* **scale** (float or Tensor) : standard deviation of the distribution.


**Returns**

Augmented states.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/gaussian_noise.py/#L22)
```python
.forward(
   x: th.Tensor
)
```

