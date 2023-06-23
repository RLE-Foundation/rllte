#


## BasePolicy
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_policy.py/#L33)
```python 
BasePolicy(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int,
   hidden_dim: int, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None,
   init_method: Callable = nn.init.orthogonal_
)
```


---
Base class for all policies.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Optional[Dict[str, Any]]) : Optimizer keyword arguments.
* **init_method** (Callable) : Initialization method.


**Returns**

Base policy instance.


**Methods:**


### .act
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_policy.py/#L91)
```python
.act(
   obs: th.Tensor, training: bool = True
)
```

---
Select an action from the input observation.


**Args**

* **obs** (th.Tensor) : Observation from the environment.
* **training** (bool) : Whether the agent is being trained or not.


**Returns**

Sampled actions, estimated values, ..., depends on specific algorithms.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_policy.py/#L103)
```python
.freeze()
```

---
Freeze the policy.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_policy.py/#L107)
```python
.save(
   path: Path, pretraining: bool = False
)
```

---
Save models.


**Args**

* **path** (Path) : Save path.
* **pretraining** (bool) : Pre-training mode.


**Returns**

None.

### .load
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_policy.py/#L119)
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
