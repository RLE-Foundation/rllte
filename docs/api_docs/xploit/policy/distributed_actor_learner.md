#


## DistributedActorLearner
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L316)
```python 
DistributedActorLearner(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int,
   hidden_dim: int = 512, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None,
   init_method: Callable = nn.init.orthogonal_, use_lstm: bool = False
)
```


---
Actor network for IMPALA that supports LSTM module.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Optional[Dict[str, Any]]) : Optimizer keyword arguments.
* **init_method** (Callable) : Initialization method.
* **use_lstm** (bool) : Whether to use LSTM module.


**Returns**

Actor-Critic network.


**Methods:**


### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L371)
```python
.freeze(
   encoder: nn.Module, dist: Distribution
)
```

---
Freeze all the elements like `encoder` and `dist`.


**Args**

* **encoder** (nn.Module) : Encoder network.
* **dist** (Distribution) : Distribution class.


**Returns**

None.

### .to
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L397)
```python
.to(
   device: th.device
)
```


### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L400)
```python
.save(
   path: Path
)
```

---
Save models.


**Args**

* **path** (Path) : Save path.


**Returns**

None.

### .load
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/distributed_actor_learner.py/#L412)
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
