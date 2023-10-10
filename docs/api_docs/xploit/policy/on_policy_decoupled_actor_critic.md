#


## OnPolicyDecoupledActorCritic
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_decoupled_actor_critic.py/#L45)
```python 
OnPolicyDecoupledActorCritic(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int,
   hidden_dim: int = 512, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None, init_fn: str = 'orthogonal'
)
```


---
Actor-Critic network for on-policy algorithms like `DAAC`.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Dict[str, Any]) : Optimizer keyword arguments.
* **init_fn** (str) : Parameters initialization method.


**Returns**

Actor-Critic network instance.


**Methods:**


### .describe
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_decoupled_actor_critic.py/#L116)
```python
.describe()
```

---
Describe the policy.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_decoupled_actor_critic.py/#L130)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_decoupled_actor_critic.py/#L157)
```python
.forward(
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_decoupled_actor_critic.py/#L180)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_decoupled_actor_critic.py/#L191)
```python
.evaluate_actions(
   obs: th.Tensor, actions: th.Tensor
)
```

---
Evaluate actions according to the current policy given the observations.


**Args**

* **obs** (th.Tensor) : Sampled observations.
* **actions** (th.Tensor) : Sampled actions.


**Returns**

Estimated values, log of the probability evaluated at `actions`, entropy of distribution.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_decoupled_actor_critic.py/#L216)
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
