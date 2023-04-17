#


## MnihCnnEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/mnih_cnn_encoder.py\#L8)
```python 
MnihCnnEncoder(
   observation_space: Space, feature_dim: int = 0
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.
Proposed by Mnih V, Kavukcuoglu K, Silver D, et al. Playing atari with deep reinforcement learning[J]. arXiv preprint arXiv:1312.5602, 2013.
Target task: Atari games.


**Args**

* **observation_space** (Space) : Observation space of the environment.
* **feature_dim** (int) : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/mnih_cnn_encoder.py\#L45)
```python
.forward(
   obs: Tensor
)
```

