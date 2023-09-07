#


## OffPolicyDoubleActorDoubleCritic
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_actor_double_critic.py/#L41)
```python 
OffPolicyDoubleActorDoubleCritic(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int = 64,
   hidden_dim: int = 1024, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None, init_fn: Optional[str] = None
)
```


---
Double deterministic actor network and double critic network for off-policy algortithms like `DDPG`, `TD3`.
Here the 'self.dist' refers to an action noise instance.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Optional[Dict[str, Any]]) : Optimizer keyword arguments.
* **init_fn** (Optional[str]) : Parameters initialization method.


**Returns**

Actor-Critic network.


**Methods:**


### .describe
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_actor_double_critic.py/#L94)
```python
.describe()
```

---
Describe the policy.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_actor_double_critic.py/#L109)
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

### .explore
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_actor_double_critic.py/#L135)
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

### .forward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_actor_double_critic.py/#L146)
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

### .get_dist
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_actor_double_critic.py/#L167)
```python
.get_dist(
   obs: th.Tensor, step: int
)
```

---
Get sample distribution.


**Args**

* **obs** (th.Tensor) : Observations.
* **step** (int) : Global training step.


**Returns**

RLLTE distribution.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_actor_double_critic.py/#L184)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_double_actor_double_critic.py/#L200)
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
