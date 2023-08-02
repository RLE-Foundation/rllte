#


## OnPolicyAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/on_policy_agent.py/#L37)
```python 
OnPolicyAgent(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_steps: int = 128,
   eval_every_episodes: int = 10
)
```


---
Trainer for on-policy algorithms.


**Args**

* **env** (gym.Env) : A Gym-like environment for training.
* **eval_env** (gym.Env) : A Gym-like environment for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on pre-training model or not.
* **num_steps** (int) : The sample length of per rollout.
* **eval_every_episodes** (int) : Evaluation interval.


**Returns**

On-policy agent instance.


**Methods:**


### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/on_policy_agent.py/#L69)
```python
.update()
```

---
Update the agent. Implemented by individual algorithms.


### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/on_policy_agent.py/#L74)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/on_policy_agent.py/#L210)
```python
.eval()
```

---
Evaluation function.
