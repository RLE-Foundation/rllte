#


## TassaCnnEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/tassa_cnn_encoder.py\#L8)
```python 
TassaCnnEncoder(
   observation_space: Space, feature_dim: int = 50
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.
Proposed by Tassa Y, Doron Y, Muldal A, et al. Deepmind control suite[J]. arXiv preprint arXiv:1801.00690, 2018.
Target task: DeepMind Control Suite.


**Args**

* **observation_space** (Space) : Observation space of the environment.
* **feature_dim** (int) : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/encoder/tassa_cnn_encoder.py\#L47)
```python
.forward(
   obs: Tensor
)
```

