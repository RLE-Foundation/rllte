#


## Categorical
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/categorical.py\#L8)
```python 
Categorical(
   logits: Tensor
)
```


---
Categorical distribution for sampling actions in discrete control tasks.

**Args**

* **logits** (Tensor) : The event log probabilities (unnormalized).


**Returns**

Categorical distribution instance.


**Methods:**


### .logits
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/categorical.py\#L26)
```python
.logits()
```

---
Returns the unnormalized log probabilities.

### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/categorical.py\#L30)
```python
.sample(
   sample_shape: TorchSize = torch.Size()
)
```

---
Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.


**Args**

* **sample_shape** (TorchSize) : The size of the sample to be drawn.


**Returns**

A sample_shape shaped sample.

### .log_prob
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/categorical.py\#L42)
```python
.log_prob(
   actions: Tensor
)
```

---
Returns the log of the probability density/mass function evaluated at `value`.


**Args**

* **actions** (Tensor) : The actions to be evaluated.


**Returns**

The log_prob value.

### .entropy
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/categorical.py\#L58)
```python
.entropy()
```

---
Returns the Shannon entropy of distribution.

### .mode
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/categorical.py\#L63)
```python
.mode()
```

---
Returns the mode of the distribution.

### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/categorical.py\#L68)
```python
.mean()
```

---
Returns the mean of the distribution.

### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/categorical.py\#L72)
```python
.reset()
```

---
Reset the distribution.

### .rsample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/distribution/categorical.py\#L76)
```python
.rsample(
   sample_shape: TorchSize = ...
)
```

