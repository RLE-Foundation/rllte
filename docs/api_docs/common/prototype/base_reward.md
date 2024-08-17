#


## BaseReward
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_reward.py/#L38)
```python 
BaseReward(
   envs: VectorEnv, device: str = 'cpu', beta: float = 1.0, kappa: float = 0.0,
   gamma: Optional[float] = None, rwd_norm_type: str = 'rms', obs_norm_type: str = 'rms'
)
```


---
Base class of reward module.


**Args**

* **envs** (VectorEnv) : The vectorized environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **beta** (float) : The initial weighting coefficient of the intrinsic rewards.
* **kappa** (float) : The decay rate of the weighting coefficient.
* **gamma** (Optional[float]) : Intrinsic reward discount rate, default is `None`.
* **rwd_norm_type** (str) : Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
* **obs_norm_type** (str) : Normalization type for observations data from ['rms', 'none'].


**Returns**

Instance of the base reward module.


**Methods:**


### .weight
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_reward.py/#L101)
```python
.weight()
```

---
Get the weighting coefficient of the intrinsic rewards.

### .scale
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_reward.py/#L105)
```python
.scale(
   rewards: th.Tensor
)
```

---
Scale the intrinsic rewards.


**Args**

* **rewards** (th.Tensor) : The intrinsic rewards with shape (n_steps, n_envs).


**Returns**

The scaled intrinsic rewards.

### .normalize
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_reward.py/#L131)
```python
.normalize(
   x: th.Tensor
)
```

---
Normalize the observations data, especially useful for images-based observations.

### .init_normalization
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_reward.py/#L142)
```python
.init_normalization()
```

---
Initialize the normalization parameters for observations if the RMS is used.

### .watch
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_reward.py/#L186)
```python
.watch(
   observations: th.Tensor, actions: th.Tensor, rewards: th.Tensor,
   terminateds: th.Tensor, truncateds: th.Tensor, next_observations: th.Tensor
)
```

---
Watch the interaction processes and obtain necessary elements for reward computation.


**Args**

* **observations** (th.Tensor) : Observations data with shape (n_envs, *obs_shape).
* **actions** (th.Tensor) : Actions data with shape (n_envs, *action_shape).
* **rewards** (th.Tensor) : Extrinsic rewards data with shape (n_envs).
* **terminateds** (th.Tensor) : Termination signals with shape (n_envs).
* **truncateds** (th.Tensor) : Truncation signals with shape (n_envs).
* **next_observations** (th.Tensor) : Next observations data with shape (n_envs, *obs_shape).


**Returns**

Feedbacks for the current samples.

### .compute
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_reward.py/#L210)
```python
.compute(
   samples: Dict[str, th.Tensor], sync: bool = True
)
```

---
Compute the rewards for current samples.


**Args**

* **samples** (Dict[str, th.Tensor]) : The collected samples. A python dict consists of multiple tensors,
    whose keys are ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'].
    For example, the data shape of 'observations' is (n_steps, n_envs, *obs_shape).
* **sync** (bool) : Whether to update the reward module after the `compute` function, default is `True`.


**Returns**

The intrinsic rewards.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_reward.py/#L241)
```python
.update(
   samples: Dict[str, th.Tensor]
)
```

---
Update the reward module if necessary.


**Args**

* **samples** (Dict[str, th.Tensor]) : The collected samples same as the `compute` function.


**Returns**

None.
