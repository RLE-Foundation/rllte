#


## ResNetEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/resnet_encoder.py/#L70)
```python 
ResNetEncoder(
   observation_space: Space, feature_dim: int = 0, net_arch: List[int] = [16, 32, 32]
)
```


---
ResNet-like encoder for processing image-based observations.


**Args**

* **observation_space**  : Observation space of the environment.
* **feature_dim**  : Number of features extracted.
* **net_arch**  : Architecture of the network.
    It represents the out channels of each residual layer.
    The length of this list is the number of residual layers.


**Returns**

ResNet-like encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/resnet_encoder.py/#L112)
```python
.forward(
   obs: Tensor
)
```

