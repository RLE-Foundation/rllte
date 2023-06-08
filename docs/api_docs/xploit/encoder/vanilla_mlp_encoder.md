#


## VanillaMlpEncoder
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/vanilla_mlp_encoder.py/#L9)
```python 
VanillaMlpEncoder(
   observation_space: gym.Space, feature_dim: int = 64, hidden_dim: int = 64
)
```


---
Multi layer perceptron (MLP) for processing state-based inputs.


**Args**

* **observation_space** (Space) : The observation space of environment.
* **feature_dim** (int) : Number of features extracted.
* **hidden_dim** (int) : Number of units per hidden layer.


**Returns**

Mlp-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/vanilla_mlp_encoder.py/#L32)
```python
.forward(
   obs: th.Tensor
)
```

