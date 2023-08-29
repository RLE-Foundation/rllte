#


## OnPolicyAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/on_policy_agent.py/#L37)
```python 
OnPolicyAgent(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_steps: int = 128
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


**Returns**

On-policy agent instance.


**Methods:**


### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/on_policy_agent.py/#L66)
```python
.update()
```

---
Update the agent. Implemented by individual algorithms.

### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/on_policy_agent.py/#L70)
```python
.train(
   num_train_steps: int, init_model_path: Optional[str] = None, log_interval: int = 1,
   eval_interval: int = 100, num_eval_episodes: int = 10, th_compile: bool = True
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/on_policy_agent.py/#L192)
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
