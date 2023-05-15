#


## RaileanuCnnEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/raileanu_cnn_encoder.py\#L11)
```python 
RaileanuCnnEncoder(
   observation_space: Union[gym.Space, DictConfig], feature_dim: int = 0
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.
Proposed by Raileanu R, Rocktäschel T. Ride: Rewarding impact-driven exploration for 
procedurally-generated environments[J]. arXiv preprint arXiv:2002.12292, 2020.
Target task: MiniGrid games.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **feature_dim** (int) : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/raileanu_cnn_encoder.py\#L52)
```python
.forward(
   obs: th.Tensor
)
```

