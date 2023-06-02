#


## GIRM
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/girm.py/#L169)
```python 
GIRM(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   beta: float = 0.05, kappa: float = 2.5e-05, latent_dim: int = 128, lr: float = 0.001,
   batch_size: int = 64, lambd: float = 0.5, lambd_recon: float = 1.0,
   lambd_action: float = 1.0, kld_loss_beta: float = 1.0
)
```


---
Intrinsic Reward Driven Imitation Learning via Generative Model (GIRM).
See paper: http://proceedings.mlr.press/v119/yu20d/yu20d.pdf


**Args**

* **observation_space** (Space) : The observation space of environment. 
* **action_space** (Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **beta** (float) : The initial weighting coefficient of the intrinsic rewards.
* **kappa** (float) : The decay rate.
* **latent_dim** (int) : The dimension of encoding vectors.
* **lr** (float) : The learning rate.
* **batch_size** (int) : The batch size for update.
* **lambd** (float) : The weighting coefficient for combining actions.
* **lambd_recon** (float) : Weighting coefficient of the reconstruction loss.
* **lambd_action** (float) : Weighting coefficient of the action loss.
* **kld_loss_beta** (float) : Weighting coefficient of the divergence loss.


**Returns**

Instance of GIRM.


**Methods:**


### .get_vae_loss
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/girm.py/#L229)
```python
.get_vae_loss(
   recon_x: th.Tensor, x: th.Tensor, mean: th.Tensor, logvar: th.Tensor
)
```

---
Compute the vae loss.


**Args**

* **recon_x** (Tensor) : Reconstructed x.
* **x** (Tensor) : Input x.
* **mean** (Tensor) : Sample mean.
* **logvar** (Tensor) : Log of the sample variance.


**Returns**

Loss values.

### .compute_irs
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/girm.py/#L246)
```python
.compute_irs(
   samples: Dict, step: int = 0
)
```

---
Compute the intrinsic rewards for current samples.


**Args**

* **samples** (Dict) : The collected samples. A python dict like
    {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
    actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
    rewards (n_steps, n_envs) <class 'th.Tensor'>,
    next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.
* **step** (int) : The global training step.


**Returns**

The intrinsic rewards.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/girm.py/#L295)
```python
.update(
   samples: Dict
)
```

---
Update the intrinsic reward module if necessary.


**Args**

* **samples**  : The collected samples. A python dict like
    {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
    actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
    rewards (n_steps, n_envs) <class 'th.Tensor'>,
    next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.


**Returns**

None
