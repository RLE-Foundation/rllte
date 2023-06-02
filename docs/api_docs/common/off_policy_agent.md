#


## OffPolicyAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/off_policy_agent.py/#L11)
```python 
OffPolicyAgent(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_init_steps: int = 2000,
   eval_every_steps: int = 5000
)
```


---
Trainer for off-policy algorithms.


**Args**

* **env** (Env) : A Gym-like environment for training.
* **eval_env** (Env) : A Gym-like environment for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on pre-training model or not.
* **num_init_steps** (int) : Number of initial exploration steps.
* **eval_every_steps** (int) : Evaluation interval.


**Returns**

Off-policy agent instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/off_policy_agent.py/#L47)
```python
.train(
   num_train_steps: int = 100000, init_model_path: Optional[str] = None
)
```

---
Training function.


**Args**

* **num_train_steps** (int) : Number of training steps.
* **init_model_path** (Optional[str]) : Path of Iinitial model parameters.


**Returns**

None.

### .eval
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/off_policy_agent.py/#L132)
```python
.eval()
```

---
Evaluation function.
