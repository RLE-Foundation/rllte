#


## NGU
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/ngu.py/#L36)
```python 
NGU(
   envs: VectorEnv, device: str = 'cpu', beta: float = 1.0, kappa: float = 0.0,
   gamma: float = None, rwd_norm_type: str = 'rms', obs_norm_type: str = 'rms',
   latent_dim: int = 32, lr: float = 0.001, batch_size: int = 256, k: int = 10,
   kernel_cluster_distance: float = 0.008, kernel_epsilon: float = 0.0001,
   c: float = 0.001, sm: float = 8.0, mrs: float = 5.0, update_proportion: float = 1.0,
   encoder_model: str = 'mnih', weight_init: str = 'default'
)
```


---
Never Give Up: Learning Directed Exploration Strategies (NGU).
See paper: https://arxiv.org/pdf/2002.06038


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
* **batch_size** (int) : The batch size for update.
* **k** (int) : Number of neighbors.
* **kernel_cluster_distance** (float) : The kernel cluster distance.
* **kernel_epsilon** (float) : The kernel constant.
* **c** (float) : The pseudo-counts constant.
* **sm** (float) : The kernel maximum similarity.
* **mrs** (float) : The maximum reward scaling.
* **update_proportion** (float) : The proportion of the training data used for updating the forward dynamics models.



**Returns**

Instance of NGU.


**Methods:**


### .compute
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/ngu.py/#L126)
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
