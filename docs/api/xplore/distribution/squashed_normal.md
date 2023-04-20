#


## SquashedNormal
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/squashed_normal.py\#L40)
```python 
SquashedNormal(
   mu: Tensor, sigma: Tensor
)
```


---
Squashed normal distribution for Soft Actor-Critic learner.


**Args**

* **mu** (Tensor) : The mean of the distribution (often referred to as mu).
* **sigma** (Tensor) : The standard deviation of the distribution (often referred to as sigma).


**Returns**

Squashed normal distribution instance.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/squashed_normal.py\#L61)
```python
.sample(
   sample_shape: TorchSize = torch.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.


**Args**

* **sample_shape** (TorchSize) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .rsample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/squashed_normal.py\#L72)
```python
.rsample(
   sample_shape: TorchSize = torch.Size()
)
```

---
Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples if the distribution parameters are batched.


**Args**

* **sample_shape** (TorchSize) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/squashed_normal.py\#L84)
```python
.mean()
```

---
Return the transformed mean.

### .log_prob
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/squashed_normal.py\#L91)
```python
.log_prob(
   actions: Tensor
)
```

---
Scores the sample by inverting the transform(s) and computing the score using the score of the base distribution and the log abs det jacobian.

**Args**

* **actions** (Tensor) : The actions to be evaluated.


**Returns**

The log_prob value.

### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/squashed_normal.py\#L101)
```python
.reset()
```

---
Reset the distribution.

### .entropy
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/squashed_normal.py\#L105)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .mode
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/squashed_normal.py\#L109)
```python
.mode()
```

---
Returns the mode of the distribution.
