#


## BaseAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L56)
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


**Returns**

Base agent instance.


**Methods:**


### .get_npu_name
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L140)
```python
.get_npu_name()
```

---
Get NPU name.

### .check
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L153)
```python
.check()
```

---
Check the compatibility of selected modules.

### .set
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L197)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L250)
```python
.mode(
   training: bool = True
)
```

---
Set the training mode.


**Args**

* **training** (bool) : True (training) or False (evaluation).


**Returns**

None.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L263)
```python
.update()
```

---
Update function of the agent.

### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L267)
```python
.train()
```

---
Training function.

### .eval
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_agent.py/#L271)
```python
.eval()
```

---
Evaluation function.
