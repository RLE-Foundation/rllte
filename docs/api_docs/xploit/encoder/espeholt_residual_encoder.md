#


## EspeholtResidualEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/espeholt_residual_encoder.py/#L68)
```python 
EspeholtResidualEncoder(
   observation_space: Union[gym.Space, DictConfig], feature_dim: int = 0,
   net_arch: List[int] = [16, 32, 32]
)
```


---
ResNet-like encoder for processing image-based observations.
Proposed by Espeholt L, Soyer H, Munos R, et al. Impala: Scalable distributed deep-rl with importance
weighted actor-learner architectures[C]//International conference on machine learning. PMLR, 2018: 1407-1416.
Target task: Atari games and Procgen games.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **feature_dim** (int) : Number of features extracted.
* **net_arch** (List) : Architecture of the network.
    It represents the out channels of each residual layer.
    The length of this list is the number of residual layers.


**Returns**

ResNet-like encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/espeholt_residual_encoder.py/#L111)
```python
.forward(
   obs: th.Tensor
)
```

