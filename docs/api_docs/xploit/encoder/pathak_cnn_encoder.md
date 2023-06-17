#


## PathakCnnEncoder
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/pathak_cnn_encoder.py/#L33)
```python 
PathakCnnEncoder(
   observation_space: gym.Space, feature_dim: int = 0
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.
Proposed by Pathak D, Agrawal P, Efros A A, et al. Curiosity-driven exploration by self-supervised prediction[C]//
International conference on machine learning. PMLR, 2017: 2778-2787.
Target task: Atari and MiniGrid games.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **feature_dim** (int) : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/pathak_cnn_encoder.py/#L72)
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
