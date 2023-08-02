#


## DistributedAgent
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/distributed_agent.py/#L42)
```python 
DistributedAgent(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', num_steps: int = 80, num_actors: int = 45, num_learners: int = 4,
   num_storages: int = 60, **kwargs
)
```


---
Trainer for distributed algorithms.


**Args**

* **env** (gym.Env) : A Gym-like environment for training.
* **eval_env** (gym.Env) : A Gym-like environment for evaluation.
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/distributed_agent.py/#L88)
```python
.run(
   env: gym.Env, actor_idx: int, free_queue: mp.SimpleQueue,
   full_queue: mp.SimpleQueue
)
```

---
Sample function of each actor. Implemented by individual algorithms.


**Args**

* **env** (gym.Env) : A Gym-like environment wrapped by `DistributedWrapper`.
* **actor_idx** (int) : The index of actor.
* **free_queue** (Queue) : Free queue for communication.
* **full_queue** (Queue) : Full queue for communication.


**Returns**

None.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/distributed_agent.py/#L138)
```python
.update()
```

---
Update the agent. Implemented by individual algorithms.

### .train
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/distributed_agent.py/#L142)
```python
.train(
   num_train_steps: int = 30000000, init_model_path: Optional[str] = None
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/distributed_agent.py/#L292)
```python
.eval()
```

---
Evaluation function.
