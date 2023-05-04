#


## BaseEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/base.py/#L20)
```python 
BaseEncoder(
   observation_space: Union[gym.Space, DictConfig], feature_dim: int = 0
)
```


---
Base class that represents a features extractor.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **feature_dim** (int) : Number of features extracted.


**Returns**

The base encoder class


**Methods:**


### .feature_dim
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/base.py/#L39)
```python
.feature_dim()
```

