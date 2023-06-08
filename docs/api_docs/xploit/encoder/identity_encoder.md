#


## IdentityEncoder
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/identity_encoder.py/#L9)
```python 
IdentityEncoder(
   observation_space: gym.Space, feature_dim: int = 64
)
```


---
Identity encoder for state-based observations.


**Args**

* **observation_space** (Space) : The observation space of environment.
* **feature_dim** (int) : Number of features extracted.


**Returns**

Identity encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/identity_encoder.py/#L28)
```python
.forward(
   obs: th.Tensor
)
```

