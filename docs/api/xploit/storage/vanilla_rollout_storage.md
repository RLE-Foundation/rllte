#


## VanillaRolloutStorage
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/vanilla_rollout_storage.py\#L7)
```python 
VanillaRolloutStorage(
   device: Device, obs_shape: Tuple, action_shape: Tuple, action_type: str,
   num_steps: int, num_envs: int, discount: float = 0.99, gae_lambda: float = 0.95
)
```


---
Vanilla rollout storage for on-policy algorithms.


**Args**

* **device** (Device) : Device (cpu, cuda, ...) on which the code should be run.
* **obs_shape** (Tuple) : The data shape of observations.
* **action_shape** (Tuple) : The data shape of actions.
* **action_type** (str) : The type of actions, 'Discrete' or 'Box'.
* **num_steps** (int) : The sample length of per rollout.
* **num_envs** (int) : The number of parallel environments.
* **discount** (float) : discount factor.
* **gae_lambda** (float) : Weighting coefficient for generalized advantage estimation (GAE).


**Returns**

Vanilla rollout storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/vanilla_rollout_storage.py\#L93)
```python
.add(
   obs: Tensor, actions: Tensor, rewards: Tensor, terminateds: Tensor,
   truncateds: Tensor, log_probs: Tensor, values: Tensor
)
```

---
Add sampled transitions into storage.


**Args**

* **obs** (Tensor) : Observations.
* **actions** (Tensor) : Actions.
* **rewards** (Tensor) : Rewards.
* **terminateds** (Tensor) : Terminateds.
* **truncateds** (Tensor) : Truncateds.
* **log_probs** (Tensor) : Log of the probability evaluated at `actions`.
* **values** (Tensor) : Estimated values.


**Returns**

None.

### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/vanilla_rollout_storage.py\#L127)
```python
.reset()
```

---
Reset the terminal state of each env.

### .compute_returns_and_advantages
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/vanilla_rollout_storage.py\#L132)
```python
.compute_returns_and_advantages(
   last_values: Tensor
)
```

---
Perform generalized advantage estimation (GAE).


**Args**

* **last_values** (Tensor) : Estimated values of the last step.
* **gamma** (float) : Discount factor.
* **gae_lamdba** (float) : Coefficient of GAE.


**Returns**

None.

### .generator
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/vanilla_rollout_storage.py\#L164)
```python
.generator(
   num_mini_batch: int = None
)
```

---
Sample data from storage.


**Args**

* **num_mini_batch** (int) : Number of mini-batches


**Returns**

Batch data.
