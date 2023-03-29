#


## Categorical
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L6)
```python 
Categorical(
   logits = None
)
```


---
Categorical distribution for sampling actions in discrete control tasks.

**Args**

* **logits**  : event log probabilities (unnormalized).


**Returns**

Categorical distribution instance.


**Methods:**


### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L17)
```python
.sample()
```


### .log_probs
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L20)
```python
.log_probs(
   actions
)
```


### .mean
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/distribution/categorical.py/#L30)
```python
.mean()
```

