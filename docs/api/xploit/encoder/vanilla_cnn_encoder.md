#


## VanillaCnnEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/vanilla_cnn_encoder.py/#L8)
```python 
VanillaCnnEncoder(
   observation_space: Space, feature_dim: int = 50
)
```


---
Convolutional neural network (CNN)-based encoder for processing image-based observations.


**Args**

* **observation_space**  : Observation space of the environment.
* **feature_dim**  : Number of features extracted.


**Returns**

CNN-based encoder instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/encoder/vanilla_cnn_encoder.py/#L44)
```python
.forward(
   obs: Tensor
)
```

