#


## IMPALA
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/impala.py/#L114)
```python 
IMPALA(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', num_steps: int = 80, num_actors: int = 45, num_learners: int = 4,
   num_storages: int = 60, feature_dim: int = 512, batch_size: int = 4, lr: float = 0.0004,
   eps: float = 0.01, hidden_dim: int = 512, use_lstm: bool = False, ent_coef: float = 0.01,
   baseline_coef: float = 0.5, max_grad_norm: float = 40, discount: float = 0.99,
   init_fn: str = 'identity'
)
```


---
Importance Weighted Actor-Learner Architecture (IMPALA).
Based on: https://github.com/facebookresearch/torchbeast/blob/main/torchbeast/monobeast.py


**Args**

* **env** (gym.Env) : A Gym-like environment for training.
* **eval_env** (gym.Env) : A Gym-like environment for evaluation.
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
* **init_fn** (str) : Parameters initialization method.



**Returns**

IMPALA agent instance.


**Methods:**


### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/impala.py/#L231)
```python
.update(
   batch: Dict, lock = threading.Lock()
)
```

---
Update the learner model.


**Args**

* **batch** (Batch) : Batch samples.
* **lock** (Lock) : Thread lock.


**Returns**

Training metrics.
