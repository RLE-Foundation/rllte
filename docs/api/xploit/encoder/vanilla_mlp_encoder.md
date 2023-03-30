#


## VanillaMlpEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/vanilla_mlp_encoder.py/#L8)
```python 
VanillaMlpEncoder(
   observation_space: Space, feature_dim: int = 64, hidden_dim: int = 256
)
```


---
Multi layer perceptron (MLP) for processing state-based inputs.


**Args**

* **observation_space**  : Observation space of the environment.
* **feature_dim**  : Number of features extracted.
* **hidden_dim**  : Number of units per hidden layer.


**Returns**

Mlp-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/vanilla_mlp_encoder.py/#L38)
```python
.forward(
   obs: Tensor
)
```

