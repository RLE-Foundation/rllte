#


## OffPolicyDoubleQNetwork
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_qnetwork.py/#L39)
```python 
OffPolicyDoubleQNetwork(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int = 64,
   hidden_dim: int = 1024, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None, init_fn: str = 'orthogonal'
)
```


---
Q-network for off-policy algortithms like `DQN`.

Structure: self.encoder (shared by actor and critic), self.qnet, self.qnet_target
Optimizers: self.opt -> (self.qnet, self.qnet_target)


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Dict[str, Any]) : Optimizer keyword arguments.
* **init_fn** (str) : Parameters initialization method.


**Returns**

Actor network instance.


**Methods:**


### .describe
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_qnetwork.py/#L92)
```python
.describe()
```

---
Describe the policy.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_qnetwork.py/#L103)
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

### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_qnetwork.py/#L123)
```python
.forward(
   obs: th.Tensor, training: bool = True, step: int = 0
)
```

---
Sample actions based on observations.


**Args**

* **obs** (th.Tensor) : Observations.
* **training** (bool) : Training mode, True or False.
* **step** (int) : Global training step.


**Returns**

Sampled actions.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_qnetwork.py/#L139)
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
