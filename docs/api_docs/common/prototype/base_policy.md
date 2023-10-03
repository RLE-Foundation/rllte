#


## BasePolicy
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_policy.py/#L37)
```python 
BasePolicy(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int,
   hidden_dim: int, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None, init_fn: str = 'orthogonal'
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
* **opt_kwargs** (Dict[str, Any]) : Optimizer keyword arguments.
* **init_fn** (str) : Parameters initialization method.


**Returns**

Base policy instance.


**Methods:**


### .optimizers
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_policy.py/#L89)
```python
.optimizers()
```

---
Get optimizers.

### .describe
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_policy.py/#L95)
```python
.describe()
```

---
Describe the policy.

### .describe
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_policy.py/#L95)
```python
.describe()
```

---
Describe the policy.

### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_policy.py/#L99)
```python
.forward(
   obs: th.Tensor, training: bool = True
)
```

---
Forward method.


**Args**

* **obs** (th.Tensor) : Observation from the environment.
* **training** (bool) : Whether the agent is being trained or not.


**Returns**

Sampled actions, estimated values, ..., depends on specific algorithms.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_policy.py/#L111)
```python
.freeze(
   *args, **kwargs
)
```

---
Freeze the policy and start training.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_policy.py/#L115)
```python
.save(
   *args, **kwargs
)
```

---
Save models.

### .load
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_policy.py/#L118)
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
