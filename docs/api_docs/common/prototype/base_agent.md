#


## BaseAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L62)
```python 
BaseAgent(
   env: VecEnv, eval_env: Optional[VecEnv] = None, tag: str = 'default', seed: int = 1,
   device: str = 'auto', pretraining: bool = False
)
```


---
Base class of the agent.


**Args**

* **env** (VecEnv) : Vectorized environments for training.
* **eval_env** (VecEnv) : Vectorized environments for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on pre-training model or not.


**Returns**

Base agent instance.


**Methods:**


### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L153)
```python
.freeze(
   **kwargs
)
```

---
Freeze the agent and get ready for training.

### .check
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L174)
```python
.check()
```

---
Check the compatibility of selected modules.

### .set
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L200)
```python
.set(
   encoder: Optional[Encoder] = None, policy: Optional[Policy] = None,
   storage: Optional[Storage] = None, distribution: Optional[Distribution] = None,
   augmentation: Optional[Augmentation] = None,
   reward: Optional[IntrinsicRewardModule] = None
)
```

---
Set a module for the agent.


**Args**

* **encoder** (Optional[Encoder]) : An encoder of `rllte.xploit.encoder` or a custom encoder.
* **policy** (Optional[Policy]) : A policy of `rllte.xploit.policy` or a custom policy.
* **storage** (Optional[Storage]) : A storage of `rllte.xploit.storage` or a custom storage.
* **distribution** (Optional[Distribution]) : A distribution of `rllte.xplore.distribution`
    or a custom distribution.
* **augmentation** (Optional[Augmentation]) : An augmentation of `rllte.xplore.augmentation`
    or a custom augmentation.
* **reward** (Optional[IntrinsicRewardModule]) : A reward of `rllte.xplore.reward` or a custom reward.


**Returns**

None.

### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L240)
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

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L252)
```python
.save()
```

---
Save the agent.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L264)
```python
.update(
   *args, **kwargs
)
```

---
Update function of the agent.

### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L268)
```python
.train(
   num_train_steps: int, init_model_path: Optional[str], log_interval: int,
   eval_interval: int, save_interval: int, num_eval_episodes: int, th_compile: bool
)
```

---
Training function.


**Args**

* **num_train_steps** (int) : The number of training steps.
* **init_model_path** (Optional[str]) : The path of the initial model.
* **log_interval** (int) : The interval of logging.
* **eval_interval** (int) : The interval of evaluation.
* **save_interval** (int) : The interval of saving model.
* **num_eval_episodes** (int) : The number of evaluation episodes.
* **th_compile** (bool) : Whether to use `th.compile` or not.


**Returns**

None.

### .eval
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_agent.py/#L294)
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
