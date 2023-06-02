#


## TassaCnnEncoder
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/tassa_cnn_encoder.py/#L11)
```python 
TassaCnnEncoder(
   observation_space: gym.Space, feature_dim: int = 50
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.
Proposed by Tassa Y, Doron Y, Muldal A, et al. Deepmind control suite[J].
arXiv preprint arXiv:1801.00690, 2018.
Target task: DeepMind Control Suite.


**Args**

* **observation_space** (Space) : The observation space of environment.
* **feature_dim** (int) : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/encoder/tassa_cnn_encoder.py/#L48)
```python
.forward(
   obs: th.Tensor
)
```

