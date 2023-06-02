#


## BaseEncoder
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_encoder.py/#L5)
```python 
BaseEncoder(
   observation_space: gym.Space, feature_dim: int = 0
)
```


---
Base class that represents a features extractor.


**Args**

* **observation_space** (Space) : The observation space of environment.
* **feature_dim** (int) : Number of features extracted.


**Returns**

The base encoder class


**Methods:**


### .feature_dim
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_encoder.py/#L23)
```python
.feature_dim()
```

