#


## TassaCnnEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/tassa_cnn_encoder.py/#L11)
```python 
TassaCnnEncoder(
   observation_space: Union[gym.Space, DictConfig], feature_dim: int = 50
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.
Proposed by Tassa Y, Doron Y, Muldal A, et al. Deepmind control suite[J].
arXiv preprint arXiv:1801.00690, 2018.
Target task: DeepMind Control Suite.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **feature_dim** (int) : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/tassa_cnn_encoder.py/#L51)
```python
.forward(
   obs: th.Tensor
)
```

