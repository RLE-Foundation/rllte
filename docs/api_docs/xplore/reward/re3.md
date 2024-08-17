#


## RE3
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/re3.py/#L35)
```python 
RE3(
   envs: VectorEnv, device: str = 'cpu', beta: float = 1.0, kappa: float = 0.0,
   gamma: float = None, rwd_norm_type: str = 'rms', obs_norm_type: str = 'rms',
   latent_dim: int = 128, storage_size: int = 1000, k: int = 5,
   average_entropy: bool = False, encoder_model: str = 'mnih',
   weight_init: str = 'orthogonal'
)
```


---
State Entropy Maximization with Random Encoders for Efficient Exploration (RE3).
See paper: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf


**Args**

* **envs** (VectorEnv) : The vectorized environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **beta** (float) : The initial weighting coefficient of the intrinsic rewards.
* **kappa** (float) : The decay rate of the weighting coefficient.
* **gamma** (Optional[float]) : Intrinsic reward discount rate, default is `None`.
* **rwd_norm_type** (str) : Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
* **obs_norm_type** (str) : Normalization type for observations data from ['rms', 'none'].
* **latent_dim** (int) : The dimension of encoding vectors.
* **storage_size** (int) : The size of the storage for random embeddings.
* **k** (int) : Use the k-th neighbors.
* **average_entropy** (bool) : Use the average of entropy estimation.
* **encoder_model** (str) : The network architecture of the encoder from ['mnih', 'pathak'].
* **weight_init** (str) : The weight initialization method from ['default', 'orthogonal'].



**Returns**

Instance of RE3.


**Methods:**


### .watch
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/re3.py/#L96)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/re3.py/#L126)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/re3.py/#L175)
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
