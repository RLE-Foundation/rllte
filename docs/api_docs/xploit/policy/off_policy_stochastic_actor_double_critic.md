#


## OffPolicyStochasticActorDoubleCritic
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stochastic_actor_double_critic.py/#L39)
```python 
OffPolicyStochasticActorDoubleCritic(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int = 64,
   hidden_dim: int = 1024, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None,
   init_method: Callable = nn.init.orthogonal_, log_std_range: Tuple = (-10, 2)
)
```


---
Stochastic actor network and double critic network for SAC.
Here the 'self.dist' refers to an sampling distribution instance.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Optional[Dict[str, Any]]) : Optimizer keyword arguments.
* **init_method** (Callable) : Initialization method.
* **log_std_range** (Tuple) : Range of log standard deviation.


**Returns**

Actor-Critic network.


**Methods:**


### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stochastic_actor_double_critic.py/#L97)
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

### .act
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stochastic_actor_double_critic.py/#L120)
```python
.act(
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stochastic_actor_double_critic.py/#L141)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stochastic_actor_double_critic.py/#L160)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/off_policy_stochastic_actor_double_critic.py/#L176)
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
