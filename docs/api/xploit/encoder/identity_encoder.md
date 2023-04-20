#


## IdentityEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/identity_encoder.py\#L7)
```python 
IdentityEncoder(
   observation_space: Space, feature_dim: int = 64
)
```


---
Identity encoder for state-based observations.


**Args**

* **observation_space** (Space) : Observation space of the environment.
* **feature_dim** (int) : Number of features extracted.


**Returns**

Identity encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/identity_encoder.py\#L25)
```python
.forward(
   obs: Tensor
)
```

