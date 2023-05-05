#


## VanillaMlpEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/vanilla_mlp_encoder.py/#L11)
```python 
VanillaMlpEncoder(
   observation_space: Union[gym.Space, DictConfig], feature_dim: int = 64,
   hidden_dim: int = 256
)
```


---
Multi layer perceptron (MLP) for processing state-based inputs.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **feature_dim** (int) : Number of features extracted.
* **hidden_dim** (int) : Number of units per hidden layer.


**Returns**

Mlp-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/vanilla_mlp_encoder.py/#L44)
```python
.forward(
   obs: th.Tensor
)
```

