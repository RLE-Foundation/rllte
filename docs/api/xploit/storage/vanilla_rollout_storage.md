#


## VanillaRolloutStorage
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_rollout_storage.py/#L7)
```python 
VanillaRolloutStorage(
   device: torch.device, obs_shape: Tuple, action_shape: Tuple, action_type: str,
   num_steps: int, num_envs: int, discount: float = 0.99, gae_lambda: float = 0.95
)
```


---
Vanilla rollout storage for on-policy algorithms.


**Args**

* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **obs_shape**  : The data shape of observations.
* **action_shape**  : The data shape of actions.
* **action_type**  : The type of actions, 'cont' or 'dis'.
* **num_steps**  : The sample steps of per rollout.
* **num_envs**  : The number of parallel environments.
* **discount**  : discount factor.
* **gae_lambda**  : Weighting coefficient for generalized advantage estimation (GAE).


**Returns**

Vanilla rollout storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_rollout_storage.py/#L67)
```python
.add(
   obs: Any, actions: Any, rewards: Any, dones: Any, log_probs: Any, values: Any
)
```

---
Add sampled transitions into storage.


**Args**

* **obs**  : Observations.
* **actions**  : Actions.
* **rewards**  : Rewards.
* **dones**  : Dones.
* **log_probs**  : Log of the probability evaluated at `actions`.
* **values**  : Estimated values.


**Returns**

None.

### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_rollout_storage.py/#L91)
```python
.reset()
```

---
Reset the terminal state of each env.



### .compute_returns_and_advantages
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_rollout_storage.py/#L98)
```python
.compute_returns_and_advantages(
   last_values: Tensor
)
```

---
Perform generalized advantage estimation (GAE).


**Args**

* **last_values**  : Estimated values of the last step.
* **gamma**  : Discount factor.
* **gae_lamdba**  : Coefficient of GAE.


**Returns**

None.

### .generator
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_rollout_storage.py/#L126)
```python
.generator(
   num_mini_batch: int = None
)
```

---
Sample data from storage.


**Args**

* **num_mini_batch**  : Number of mini-batches


**Returns**

Batch data.
