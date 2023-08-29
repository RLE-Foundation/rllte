#


## OffPolicyStochActorDoubleCritic
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stoch_actor_double_critic.py/#L38)
```python 
OffPolicyStochActorDoubleCritic(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int = 64,
   hidden_dim: int = 1024, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None, log_std_range: Tuple = (-10, 2),
   init_fn: Optional[str] = None
)
```


---
Stochastic actor network and double critic network for off-policy algortithms like `SAC`.
Here the 'self.dist' refers to an sampling distribution instance.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Optional[Dict[str, Any]]) : Optimizer keyword arguments.
* **log_std_range** (Tuple) : Range of log standard deviation.
* **init_fn** (Optional[str]) : Parameters initialization method.


**Returns**

Actor-Critic network.


**Methods:**


### .describe
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stoch_actor_double_critic.py/#L91)
```python
.describe()
```

---
Describe the policy.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stoch_actor_double_critic.py/#L106)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stoch_actor_double_critic.py/#L131)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stoch_actor_double_critic.py/#L142)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stoch_actor_double_critic.py/#L163)
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

Action distribution.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stoch_actor_double_critic.py/#L182)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stoch_actor_double_critic.py/#L198)
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
