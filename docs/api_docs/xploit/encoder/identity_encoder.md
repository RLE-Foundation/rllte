#


## IdentityEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/identity_encoder.py/#L11)
```python 
IdentityEncoder(
   observation_space: Union[gym.Space, DictConfig], feature_dim: int = 64
)
```


---
Identity encoder for state-based observations.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **feature_dim** (int) : Number of features extracted.


**Returns**

Identity encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/identity_encoder.py/#L31)
```python
.forward(
   obs: th.Tensor
)
```

