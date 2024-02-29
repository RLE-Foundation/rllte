#


## OffPolicyAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/off_policy_agent.py/#L38)
```python 
OffPolicyAgent(
   env: VecEnv, eval_env: Optional[VecEnv] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_init_steps: int = 2000, **kwargs
)
```


---
Trainer for off-policy algorithms.


**Args**

* **env** (VecEnv) : Vectorized environments for training.
* **eval_env** (Optional[VecEnv]) : Vectorized environments for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on pre-training model or not.
* **num_init_steps** (int) : Number of initial exploration steps.
* **kwargs**  : Arbitrary arguments such as `batch_size` and `hidden_dim`.


**Returns**

Off-policy agent instance.


**Methods:**


### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/off_policy_agent.py/#L72)
```python
.update()
```

---
Update the agent. Implemented by individual algorithms.

### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/off_policy_agent.py/#L76)
```python
.train(
   num_train_steps: int, init_model_path: Optional[str] = None, log_interval: int = 1,
   eval_interval: int = 5000, save_interval: int = 5000, num_eval_episodes: int = 10,
   th_compile: bool = False, anneal_lr: bool = False
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
* **anneal_lr** (bool) : Whether to anneal the learning rate or not.


**Returns**

None.

### .eval
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/off_policy_agent.py/#L205)
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
