#


## RND
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rnd.py/#L38)
```python 
RND(
   envs: VectorEnv, device: str = 'cpu', beta: float = 1.0, kappa: float = 0.0,
   gamma: Optional[float] = None, rwd_norm_type: str = 'rms', obs_norm_type: str = 'rms',
   latent_dim: int = 128, lr: float = 0.001, batch_size: int = 256,
   update_proportion: float = 1.0, encoder_model: str = 'mnih',
   weight_init: str = 'orthogonal'
)
```


---
Exploration by Random Network Distillation (RND).
See paper: https://arxiv.org/pdf/1810.12894.pdf


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
* **update_proportion** (float) : The proportion of the training data used for updating the forward dynamics models.
* **encoder_model** (str) : The network architecture of the encoder from ['mnih', 'pathak'].
* **weight_init** (str) : The weight initialization method from ['default', 'orthogonal'].



**Returns**

Instance of RND.


**Methods:**


### .compute
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rnd.py/#L102)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rnd.py/#L138)
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
