#


## MnihCnnEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/mnih_cnn_encoder.py/#L11)
```python 
MnihCnnEncoder(
   observation_space: Union[gym.Space, DictConfig], feature_dim: int = 0
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.
Proposed by Mnih V, Kavukcuoglu K, Silver D, et al. Playing atari with
deep reinforcement learning[J]. arXiv preprint arXiv:1312.5602, 2013.
Target task: Atari games.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **feature_dim** (int) : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/mnih_cnn_encoder.py/#L49)
```python
.forward(
   obs: th.Tensor
)
```

