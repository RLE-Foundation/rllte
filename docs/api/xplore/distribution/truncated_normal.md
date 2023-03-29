#


## TruncatedNormal
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/distribution/truncated_normal.py/#L7)
```python 
TruncatedNormal(
   mu: Tensor, sigma: Tensor, low: float = -1.0, high: float = 1.0, eps: float = 1e-06
)
```


---
Truncated normal distribution for sampling noise.


**Args**

* **mu**  : Mean of the distribution.
* **sigma**  : Standard deviation of the distribution.
* **low**  : Lower bound for action range.
* **high**  : Upper bound for action range.
* **eps**  : A constant for clamping.


**Returns**

Truncated normal distribution instance.


**Methods:**


### .sample
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/distribution/truncated_normal.py/#L29)
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
