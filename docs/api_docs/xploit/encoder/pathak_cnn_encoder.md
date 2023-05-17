#


## PathakCnnEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/pathak_cnn_encoder.py\#L11)
```python 
PathakCnnEncoder(
   observation_space: Union[gym.Space, DictConfig], feature_dim: int = 0
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.
Proposed by Pathak D, Agrawal P, Efros A A, et al. Curiosity-driven exploration by self-supervised prediction[C]//
International conference on machine learning. PMLR, 2017: 2778-2787.
Target task: Atari and MiniGrid games.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **feature_dim** (int) : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/pathak_cnn_encoder.py\#L52)
```python
.forward(
   obs: th.Tensor
)
```

