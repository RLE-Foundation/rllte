#


## OnPolicySharedActorCritic
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L43)
```python 
OnPolicySharedActorCritic(
   observation_space: gym.Space, action_space: gym.Space, feature_dim: int,
   hidden_dim: int = 512, opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
   opt_kwargs: Optional[Dict[str, Any]] = None, aux_critic: bool = False,
   init_fn: str = 'orthogonal'
)
```


---
Actor-Critic network for on-policy algorithms like `PPO` and `A2C`.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **feature_dim** (int) : Number of features accepted.
* **hidden_dim** (int) : Number of units per hidden layer.
* **opt_class** (Type[th.optim.Optimizer]) : Optimizer class.
* **opt_kwargs** (Dict[str, Any]) : Optimizer keyword arguments.
* **aux_critic** (bool) : Use auxiliary critic or not, for `PPG` agent.
* **init_fn** (str) : Parameters initialization method.


**Returns**

Actor-Critic network instance.


**Methods:**


### .describe
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L111)
```python
.describe()
```

---
Describe the policy.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L126)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L147)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L170)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L181)
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

### .get_policy_outputs
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L200)
```python
.get_policy_outputs(
   obs: th.Tensor
)
```

---
Get policy outputs for training.


**Args**

* **obs** (Tensor) : Observations.


**Returns**

Policy outputs like unnormalized probabilities for `Discrete` tasks.

### .get_dist_and_aux_value
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L213)
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

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/policy/on_policy_shared_actor_critic.py/#L228)
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
