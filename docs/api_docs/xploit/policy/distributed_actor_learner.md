#


## DistributedActorLearner
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L43)
```python 
DistributedActorLearner(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int,
   hidden_dim: int = 512, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None, init_fn: str = 'orthogonal',
   use_lstm: bool = False
)
```


---
Actor-Learner network for IMPALA.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Dict[str, Any]) : Optimizer keyword arguments.
* **init_fn** (str) : Parameters initialization method.
* **use_lstm** (bool) : Whether to use LSTM module.


**Returns**

Actor-Critic network.


**Methods:**


### .describe
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L100)
```python
.describe()
```

---
Describe the policy.

### .explore
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L112)
```python
.explore(
   obs: th.Tensor
)
```

---
Explore the environment and randomly generate actions.


**Args**

* **obs** (th.Tensor) : Observation from the environment.


**Returns**

Sampled actions.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L123)
```python
.freeze(
   encoder: nn.Module, dist: Distribution
)
```

---
Freeze all the elements like `encoder` and `dist`.


**Args**

* **encoder** (nn.Module) : Encoder network.
* **dist** (Distribution) : Distribution.


**Returns**

None.

### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L151)
```python
.forward(
   *args
)
```

---
Only for inference.

### .to
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L154)
```python
.to(
   device: th.device
)
```

---
Only move the learner to device, and keep actor in CPU.


**Args**

* **device** (th.device) : Device to use.


**Returns**

None.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L165)
```python
.save(
   path: Path, pretraining: bool, global_step: int
)
```

---
Save models.


**Args**

* **path** (Path) : Save path.
* **pretraining** (bool) : Pre-training mode.
* **global_step** (int) : Global training step.


**Returns**

None.

### .load
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L179)
```python
.load(
   path: str, device: th.device
)
```

---
Load initial parameters.


**Args**

* **path** (str) : Import path.
* **device** (th.device) : Device to use.


**Returns**

None.
