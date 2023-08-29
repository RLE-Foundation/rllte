#


## BaseAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L61)
```python 
BaseAgent(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'auto', pretraining: bool = False
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


### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L147)
```python
.freeze(
   **kwargs
)
```

---
Freeze the agent and get ready for training.

### .check
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L169)
```python
.check()
```

---
Check the compatibility of selected modules.

### .set
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L204)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L257)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L270)
```python
.update()
```

---
Update function of the agent.

### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L274)
```python
.train(
   num_train_steps: int, init_model_path: Optional[str], log_interval: int,
   eval_interval: int, num_eval_episodes: int, th_compile: bool
)
```

---
Training function.


**Args**

* **num_train_steps** (int) : The number of training steps.
* **init_model_path** (Optional[str]) : The path of the initial model.
* **log_interval** (int) : The interval of logging.
* **eval_interval** (int) : The interval of evaluation.
* **num_eval_episodes** (int) : The number of evaluation episodes.
* **th_compile** (bool) : Whether to use `th.compile` or not.


**Returns**

None.

### .eval
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L298)
```python
.eval(
   num_eval_episodes: int
)
```

---
Evaluation function.


**Args**

* **num_eval_episodes** (int) : The number of evaluation episodes.


**Returns**

The evaluation results.
