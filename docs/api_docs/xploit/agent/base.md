#


## BaseAgent
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/base.py\#L9)
```python 
BaseAgent(
   observation_space: Union[gym.Space, DictConfig], action_space: Union[gym.Space,
   DictConfig], device: str, feature_dim: int, lr: float, eps: float
)
```


---
Base class of agent.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **action_space** (Space or DictConfig) : The action space of environment. When invoked by Hydra,
    'action_space' is a 'DictConfig' like
    {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.


**Returns**

Base agent instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/base.py\#L74)
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

### .act
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/base.py\#L86)
```python
.act(
   obs: th.Tensor, training: bool = True, step: int = 0
)
```

---
Sample actions based on observations.


**Args**

* **obs** (Tensor) : Observations.
* **training** (bool) : training mode, True or False.
* **step** (int) : Global training step.


**Returns**

Sampled actions.

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/base.py\#L99)
```python
.update(
   *kwargs
)
```

---
Update agent and return training metrics such as loss functions.
