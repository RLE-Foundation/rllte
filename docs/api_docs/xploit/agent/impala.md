#


## IMPALA
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/impala.py/#L89)
```python 
IMPALA(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', num_steps: int = 80, num_actors: int = 45, num_learners: int = 4,
   num_storages: int = 60, feature_dim: int = 512, batch_size: int = 4, lr: float = 0.0004,
   eps: float = 0.01, use_lstm: bool = False, ent_coef: float = 0.01,
   baseline_coef: float = 0.5, max_grad_norm: float = 40, discount: float = 0.99,
   network_init_method: str = 'identity'
)
```


---
Importance Weighted Actor-Learner Architecture (IMPALA).
Based on: https://github.com/facebookresearch/torchbeast/blob/main/torchbeast/monobeast.py


**Args**

* **env** (Env) : A Gym-like environment for training.
* **eval_env** (Env) : A Gym-like environment for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **num_steps** (int) : The sample length of per rollout.
* **num_actors** (int) : Number of actors.
* **num_learners** (int) : Number of learners.
* **num_storages** (int) : Number of storages.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **batch_size** (int) : Number of samples per batch to load.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.
* **hidden_dim** (int) : The size of the hidden layers.
* **use_lstm** (bool) : Use LSTM in the policy network or not.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **baseline_coef** (float) : Weighting coefficient of baseline value loss.
* **max_grad_norm** (float) : Maximum norm of gradients.
* **discount** (float) : Discount factor.
* **network_init_method** (str) : Network initialization method name.



**Returns**

IMPALA agent instance.


**Methods:**


### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/impala.py/#L212)
```python
.mode(
   training: bool = True
)
```

---
Set the training mode.


**Args**

* **training** (bool) : True (training) or False (testing).


**Returns**

None.

### .set
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/impala.py/#L225)
```python
.set(
   encoder: Optional[Any] = None, storage: Optional[Any] = None,
   distribution: Optional[Any] = None, augmentation: Optional[Any] = None,
   reward: Optional[Any] = None
)
```

---
Set a module for the agent.


**Args**

* **encoder** (Optional[Any]) : An encoder of `rllte.xploit.encoder` or a custom encoder.
* **storage** (Optional[Any]) : A storage of `rllte.xploit.storage` or a custom storage.
* **distribution** (Optional[Any]) : A distribution of `rllte.xplore.distribution` or a custom distribution.
* **augmentation** (Optional[Any]) : An augmentation of `rllte.xplore.augmentation` or a custom augmentation.
* **reward** (Optional[Any]) : A reward of `rllte.xplore.reward` or a custom reward.


**Returns**

None.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/impala.py/#L255)
```python
.freeze()
```

---
Freeze the structure of the agent.

### .act
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/impala.py/#L278)
```python
.act(
   env: Environment, actor_idx: int, free_queue: mp.SimpleQueue,
   full_queue: mp.SimpleQueue, init_actor_state_storages: List[th.Tensor]
)
```

---
Sampling function for each actor.


**Args**

* **env** (Environment) : A Gym-like environment wrapped by `Environment`.
* **actor_idx** (int) : The index of actor.
* **free_queue** (Queue) : Free queue for communication.
* **full_queue** (Queue) : Full queue for communication.
* **init_actor_state_storages** (List[Tensor]) : Initial states for LSTM.


**Returns**

None.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/impala.py/#L338)
```python
.update(
   batch: Dict, init_actor_states: Tuple[th.Tensor, ...],
   lr_scheduler: th.optim.lr_scheduler, lock = threading.Lock()
)
```

---
Update the learner model.


**Args**

* **batch** (Batch) : Batch samples.
* **init_actor_states** (List[Tensor]) : Initial states for LSTM.
* **lr_scheduler** (th.optim.lr_scheduler) : Learning rate scheduler.
* **lock** (Lock) : Thread lock.


**Returns**

Training metrics.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/impala.py/#L401)
```python
.save()
```

---
Save models.

### .load
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/impala.py/#L409)
```python
.load(
   path: str
)
```

---
Load initial parameters.


**Args**

* **path** (str) : Import path.


**Returns**

None.
