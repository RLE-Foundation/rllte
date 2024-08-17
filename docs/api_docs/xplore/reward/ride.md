#


## RIDE
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/ride.py/#L40)
```python 
RIDE(
   envs: VectorEnv, device: str = 'cpu', beta: float = 1.0, kappa: float = 0.0,
   gamma: float = None, rwd_norm_type: str = 'rms', obs_norm_type: str = 'none',
   latent_dim: int = 128, lr: float = 0.001, batch_size: int = 256, k: int = 10,
   kernel_cluster_distance: float = 0.008, kernel_epsilon: float = 0.0001,
   c: float = 0.001, sm: float = 8.0, update_proportion: float = 1.0,
   encoder_model: str = 'mnih', weight_init: str = 'orthogonal'
)
```


---
RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments.
See paper: https://arxiv.org/pdf/2002.12292


**Args**

* **envs** (VectorEnv) : The vectorized environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **beta** (float) : The initial weighting coefficient of the intrinsic rewards.
* **kappa** (float) : The decay rate of the weighting coefficient.
* **gamma** (Optional[float]) : Intrinsic reward discount rate, default is `None`.
* **rwd_norm_type** (str) : Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
* **obs_norm_type** (str) : Normalization type for observations data from ['rms', 'none'].
* **latent_dim** (int) : The dimension of encoding vectors.
* **lr** (float) : The learning rate.
* **batch_size** (int) : The batch size for training.
* **k** (int) : Number of neighbors.
* **kernel_cluster_distance** (float) : The kernel cluster distance.
* **kernel_epsilon** (float) : The kernel constant.
* **c** (float) : The pseudo-counts constant.
* **sm** (float) : The kernel maximum similarity.
* **update_proportion** (float) : The proportion of the training data used for updating the forward dynamics models.
* **encoder_model** (str) : The network architecture of the encoder from ['mnih', 'pathak'].
* **weight_init** (str) : The weight initialization method from ['default', 'orthogonal'].



**Returns**

Instance of RIDE.


**Methods:**


### .pseudo_counts
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/ride.py/#L138)
```python
.pseudo_counts(
   embeddings: th.Tensor, memory: List[th.Tensor]
)
```

---
Pseudo counts.


**Args**

* **embeddings** (th.Tensor) : Encoded observations.
* **memory** (List[th.Tensor]) : Episodic memory.


**Returns**

Conut values.

### .watch
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/ride.py/#L163)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/ride.py/#L206)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/ride.py/#L280)
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
