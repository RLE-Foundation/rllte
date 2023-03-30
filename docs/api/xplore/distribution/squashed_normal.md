#


## SquashedNormal
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/squashed_normal.py/#L38)
```python 
SquashedNormal(
   mu: Tensor, sigma: Tensor, low: float = -1.0, high: float = 1.0, eps: float = 1e-06
)
```


---
Squashed normal distribution for Soft Actor-Critic.


**Args**

* **mu**  : Mean of the distribution.
* **sigma**  : Standard deviation of the distribution.
* **low**  : Lower bound for action range.
* **high**  : Upper bound for action range.
* **eps**  : A constant for clamping.


**Returns**

Squashed normal distribution instance.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/squashed_normal.py/#L82)
```python
.sample(
   sample_shape = torch.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.

### .rsample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/squashed_normal.py/#L86)
```python
.rsample(
   sample_shape = torch.Size()
)
```

---
Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples if the distribution parameters are batched.

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/squashed_normal.py/#L91)
```python
.mean()
```

---
Return the transformed mean.
