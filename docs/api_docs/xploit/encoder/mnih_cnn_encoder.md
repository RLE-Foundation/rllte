#


## MnihCnnEncoder
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/mnih_cnn_encoder.py/#L33)
```python 
MnihCnnEncoder(
   observation_space: gym.Space, feature_dim: int = 0
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.
Proposed by Mnih V, Kavukcuoglu K, Silver D, et al. Playing atari with
deep reinforcement learning[J]. arXiv preprint arXiv:1312.5602, 2013.
Target task: Atari games.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **feature_dim** (int) : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/mnih_cnn_encoder.py/#L70)
```python
.forward(
   obs: th.Tensor
)
```

---
Forward method implementation.


**Args**

* **obs** (th.Tensor) : Observation tensor.


**Returns**

Encoded observation tensor.
