#


## IdentityEncoder
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/identity_encoder.py/#L33)
```python 
IdentityEncoder(
   observation_space: gym.Space, feature_dim: int = 64
)
```


---
Identity encoder for state-based observations.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **feature_dim** (int) : Number of features extracted.


**Returns**

Identity encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/identity_encoder.py/#L53)
```python
.forward(
   obs: th.Tensor
)
```

---
Forward method implementation.


**Args**

* **obs** (th.Tensor) : Observation tensor.


**Returns**

Encoded observation tensor.
