#


## BaseLearner
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/base.py\#L6)
```python 
BaseLearner(
   observation_space: Dict, action_space: Dict, device: Device, feature_dim: int,
   lr: float, eps: float
)
```


---
Base class of learner.


**Args**

* **observation_space** (Dict) : Observation space of the environment.
    For supporting Hydra, the original 'observation_space' is transformed into a dict like {"shape": observation_space.shape, }.
* **action_space** (Dict) : Action shape of the environment.
    For supporting Hydra, the original 'action_space' is transformed into a dict like
    {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
* **device** (Device) : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.


**Returns**

Base learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/base.py\#L49)
```python
.train(
   training: bool = True
)
```

---
Set the train mode.


**Args**

* **training** (bool) : True (training) or False (testing).


**Returns**

None.

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/base.py\#L61)
```python
.update(
   *kwargs
)
```

---
Update learner.


**Args**

Any possible arguments.



**Returns**

Training metrics such as loss functions.
