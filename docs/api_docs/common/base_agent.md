#


## BaseAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L54)
```python 
BaseAgent(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False
)
```


---
Base class of the agent.


**Args**

* **env** (gym.Env) : A Gym-like environment for training.
* **eval_env** (gym.Env) : A Gym-like environment for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on pre-training model or not.
* **feature_dim** (int) : Number of features extracted by the encoder.


**Returns**

Base agent instance.


**Methods:**


### .get_env_info
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L133)
```python
.get_env_info(
   env: gym.Env
)
```

---
Get the environment information.


**Args**

* **env** (gym.Env) : A Gym-like environment for training.


**Returns**

None.

### .get_npu_name
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L167)
```python
.get_npu_name()
```

---
Get NPU name.

### .check
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L180)
```python
.check()
```

---
Check the compatibility of selected modules.

### .set
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L208)
```python
.set(
   encoder: Optional[Any] = None, policy: Optional[Any] = None,
   storage: Optional[Any] = None, distribution: Optional[Any] = None,
   augmentation: Optional[Any] = None, reward: Optional[Any] = None
)
```

---
Set a module for the agent.


**Args**

* **encoder** (Optional[Any]) : An encoder of `rllte.xploit.encoder` or a custom encoder.
* **policy** (Optional[Any]) : A policy of `rllte.xploit.policy` or a custom policy.
* **storage** (Optional[Any]) : A storage of `rllte.xploit.storage` or a custom storage.
* **distribution** (Optional[Any]) : A distribution of `rllte.xplore.distribution` or a custom distribution.
* **augmentation** (Optional[Any]) : An augmentation of `rllte.xplore.augmentation` or a custom augmentation.
* **reward** (Optional[Any]) : A reward of `rllte.xplore.reward` or a custom reward.


**Returns**

None.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L258)
```python
.mode()
```

---
Set the training mode.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L262)
```python
.update()
```

---
Update the agent.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L266)
```python
.freeze()
```

---
Freeze the structure of the agent.

### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L270)
```python
.train()
```

---
Training function.

### .eval
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L274)
```python
.eval()
```

---
Evaluation function.
