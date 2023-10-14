#


## RaffinCombinedEncoder
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/raffin_combined_encoder.py/#L37)
```python 
RaffinCombinedEncoder(
   observation_space: gym.Space, feature_dim: int = 256, cnn_output_dim: int = 256
)
```


---
Combined features extractor for Dict observation spaces.
Based on: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L231


**Args**

* **observation_space** (gym.Space) : Observation space.
* **feature_dim** (int) : Number of features extracted.
* **cnn_output_dim** (int) : Number of features extracted by the CNN.


**Returns**

Identity encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/raffin_combined_encoder.py/#L69)
```python
.forward(
   obs: Dict[str, th.Tensor]
)
```

---
Forward method implementation.


**Args**

* **obs** (Dict[str, th.Tensor]) : Observation tensor.


**Returns**

Encoded observation tensor.
