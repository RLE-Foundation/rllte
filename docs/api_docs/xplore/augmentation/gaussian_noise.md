#


## GaussianNoise
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/augmentation/gaussian_noise.py/#L32)
```python 
GaussianNoise(
   mu: float = 0, sigma: float = 1.0
)
```


---
Gaussian noise operation for processing state-based observations.


**Args**

* **mu** (float or th.Tensor) : mean of the distribution.
* **scale** (float or th.Tensor) : standard deviation of the distribution.


**Returns**

Augmented states.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/augmentation/gaussian_noise.py/#L47)
```python
.forward(
   x: th.Tensor
)
```

