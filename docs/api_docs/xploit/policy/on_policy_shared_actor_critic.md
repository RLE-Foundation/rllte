#


## OnPolicySharedActorCritic
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L212)
```python 
OnPolicySharedActorCritic(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int,
   hidden_dim: int, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None,
   init_method: Callable = nn.init.orthogonal_, aux_critic: bool = False
)
```


---
Actor-Critic network using a shared encoder for on-policy algorithms like `PPO`.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Optional[Dict[str, Any]]) : Optimizer keyword arguments.
* **init_method** (Callable) : Initialization method.
* **aux_critic** (bool) : Use auxiliary critic or not, for `PPG` agent.


**Returns**

Actor-Critic network instance.


**Methods:**


### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L279)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L300)
```python
.act(
   obs: th.Tensor, training: bool = True
)
```

---
Get actions and estimated values for observations.


**Args**

* **obs** (th.Tensor) : Observations.
* **training** (bool) : training mode, `True` or `False`.


**Returns**

Sampled actions, estimated values, and log of probabilities for observations when `training` is `True`,
else only deterministic actions.

### .get_value
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L323)
```python
.get_value(
   obs: th.Tensor
)
```

---
Get estimated values for observations.


**Args**

* **obs** (th.Tensor) : Observations.


**Returns**

Estimated values.

### .evaluate_actions
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L334)
```python
.evaluate_actions(
   obs: th.Tensor, actions: th.Tensor = None
)
```

---
Evaluate actions according to the current policy given the observations.


**Args**

* **obs** (th.Tensor) : Sampled observations.
* **actions** (th.Tensor) : Sampled actions.


**Returns**

Estimated values, log of the probability evaluated at `actions`, entropy of distribution.

### .get_dist_and_aux_value
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L353)
```python
.get_dist_and_aux_value(
   obs: th.Tensor
)
```

---
Get probs and auxiliary estimated values for auxiliary phase update.


**Args**

* **obs**  : Sampled observations.


**Returns**

Sample distribution, estimated values, auxiliary estimated values.

### .get_policy_outputs
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L368)
```python
.get_policy_outputs(
   obs: th.Tensor
)
```

---
Get policy outputs for training.


**Args**

* **obs** (th.Tensor) : Observations.


**Returns**

Policy outputs like unnormalized probabilities for `Discrete` tasks.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L381)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L397)
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
