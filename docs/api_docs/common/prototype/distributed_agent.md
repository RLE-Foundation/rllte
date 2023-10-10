#


## DistributedAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/distributed_agent.py/#L42)
```python 
DistributedAgent(
   env: VecEnv, eval_env: Optional[VecEnv] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', num_steps: int = 80, num_actors: int = 45, num_learners: int = 4,
   num_storages: int = 60, **kwargs
)
```


---
Trainer for distributed algorithms.


**Args**

* **env** (VecEnv) : Vectorized environments for training.
* **eval_env** (VecEnv) : Vectorized environments for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on pre-training model or not.
* **num_steps** (int) : The sample length of per rollout.
* **num_actors** (int) : Number of actors.
* **num_learners** (int) : Number of learners.
* **num_storages** (int) : Number of storages.
* **kwargs**  : Arbitrary arguments such as `batch_size` and `hidden_dim`.


**Returns**

Distributed agent instance.


**Methods:**


### .run
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/distributed_agent.py/#L113)
```python
.run(
   env: DistributedWrapper, actor_idx: int
)
```

---
Sample function of each actor. Implemented by individual algorithms.


**Args**

* **env** (DistributedWrapper) : A Gym-like environment wrapped by `DistributedWrapper`.
* **actor_idx** (int) : The index of actor.


**Returns**

None.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/distributed_agent.py/#L155)
```python
.update(
   *args, **kwargs
)
```

---
Update the agent. Implemented by individual algorithms.

### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/distributed_agent.py/#L159)
```python
.train(
   num_train_steps: int, init_model_path: Optional[str] = None, log_interval: int = 1,
   eval_interval: int = 5000, save_interval: int = 5000, num_eval_episodes: int = 10,
   th_compile: bool = False
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/distributed_agent.py/#L287)
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
