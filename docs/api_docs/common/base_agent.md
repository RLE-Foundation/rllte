#


## BaseAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L22)
```python 
BaseAgent(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False
)
```


---
Base class of the agent.


**Args**

* **env** (Env) : A Gym-like environment for training.
* **eval_env** (Env) : A Gym-like environment for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on pre-training model or not.


**Returns**

Base agent instance.


**Methods:**


### .get_env_info
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L98)
```python
.get_env_info(
   env: gym.Env
)
```

---
Get the environment information.


**Args**

* **env** (Env) : A Gym-like environment for training.


**Returns**

None.

### .get_npu_name
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L132)
```python
.get_npu_name()
```

---
Get NPU name.

### .check
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L145)
```python
.check()
```

---
Check the compatibility of selected modules.

### .set
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L173)
```python
.set(
   encoder: Optional[Any] = None, storage: Optional[Any] = None,
   distribution: Optional[Any] = None, augmentation: Optional[Any] = None,
   reward: Optional[Any] = None
)
```

---
Set a module for the agent.


**Args**

* **encoder** (Optional[Any]) : An encoder of `rllte.xploit.encoder` or a custom encoder.
* **storage** (Optional[Any]) : A storage of `rllte.xploit.storage` or a custom storage.
* **distribution** (Optional[Any]) : A distribution of `rllte.xplore.distribution` or a custom distribution.
* **augmentation** (Optional[Any]) : An augmentation of `rllte.xplore.augmentation` or a custom augmentation.
* **reward** (Optional[Any]) : A reward of `rllte.xplore.reward` or a custom reward.


**Returns**

None.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L209)
```python
.mode()
```

---
Set the training mode.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L213)
```python
.freeze()
```

---
Freeze the structure of the agent.

### .act
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L217)
```python
.act()
```

---
Sample actions based on observations.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L221)
```python
.update()
```

---
Update the agent.

### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L225)
```python
.train()
```

---
Training function.

### .eval
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L229)
```python
.eval()
```

---
Evaluation function.

### .load
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L233)
```python
.load()
```

---
Load initial model parameters.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L237)
```python
.save()
```

---
Save the trained model.
