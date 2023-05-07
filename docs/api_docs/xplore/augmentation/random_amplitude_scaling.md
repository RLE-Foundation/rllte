#


## RandomAmplitudeScaling
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_amplitude_scaling.py/#L7)
```python 
RandomAmplitudeScaling(
   low: float = 0.6, high: float = 1.2
)
```


---
Random amplitude scaling operation for processing state-based observations.


**Args**

* **low** (float) : lower range (inclusive).
* **high** (float) : upper range (exclusive).


**Returns**

Augmented states.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_amplitude_scaling.py/#L22)
```python
.forward(
   x: th.Tensor
)
```

