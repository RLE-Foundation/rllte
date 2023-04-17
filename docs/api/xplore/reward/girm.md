#


## GIRM
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L245)
```python 
GIRM(
   obs_shape: Tuple, action_shape: Tuple, action_type: str, device: torch.device,
   beta: float, kappa: float, latent_dim: int, lr: float, batch_size: int,
   lambd: float
)
```


---
Intrinsic Reward Driven Imitation Learning via Generative Model (GIRM).
See paper: http://proceedings.mlr.press/v119/yu20d/yu20d.pdf


**Args**

* **obs_shape**  : Data shape of observation.
* **action_space**  : Data shape of action.
* **action_type**  : Continuous or discrete action. "cont" or "dis".
* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **beta**  : The initial weighting coefficient of the intrinsic rewards.
* **kappa**  : The decay rate.
* **latent_dim**  : The dimension of encoding vectors of the observations.
* **lr**  : The learning rate of inverse and forward dynamics model.
* **batch_size**  : The batch size to train the dynamic models.
* **lambd**  : The weighting coefficient for combining actions.


**Returns**

Instance of GIRM.


**Methods:**


### .compute_irs
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L339)
```python
.compute_irs(
   rollouts: Dict, step: int
)
```

---
Compute the intrinsic rewards using the collected observations.


**Args**

* **rollouts**  : The collected experiences. A python dict like
    {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
    actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
    rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.
* **step**  : The current time step.


**Returns**

The intrinsic rewards

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L412)
```python
.update(
   rollouts: Dict, lambda_recon: float = 1.0, lambda_action: float = 1.0,
   kld_loss_beta: float = 1.0
)
```

---
Update the intrinsic reward module if necessary.


**Args**

* **rollouts**  : The collected experiences. A python dict like
    {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
    actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
    rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.
* **lambda_recon**  : Weighting coefficient of the reconstruction loss.
* **lambda_action**  : Weighting coefficient of the action loss.
* **kld_loss_beta**  : Weighting coefficient of the divergence loss.


**Returns**

None

----


## CnnEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L66)
```python 
CnnEncoder(
   obs_shape: Tuple
)
```


---
CNN-based encoder of VAE.


**Args**

* **obs_shape**  : The data shape of observations.


**Returns**

CNN-based encoder.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L96)
```python
.forward(
   obs, next_obs
)
```


----


## CnnDecoder
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L119)
```python 
CnnDecoder(
   obs_shape: Tuple, action_dim: int, latent_dim: int
)
```


---
CNN-based decoder of VAE.


**Args**

* **obs_shape**  : The data shape of observations.


**Returns**

CNN-based decoder.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L163)
```python
.forward(
   z, obs
)
```


----


## MlpEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L12)
```python 
MlpEncoder(
   obs_shape: Tuple, latent_dim: int
)
```


---
MLP-based encoder of VAE.


**Args**

* **obs_shape**  : The data shape of observations.
* **latent_dim**  : The dimension of encoding vectors of the observations.


**Returns**

MLP-based encoder.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L34)
```python
.forward(
   obs: Tensor, next_obs: Tensor
)
```


----


## MlpDecoder
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L39)
```python 
MlpDecoder(
   obs_shape: Tuple, action_dim: int
)
```


---
MLP-based decoder of VAE.


**Args**

* **obs_shape**  : The data shape of observations.
* **latent_dim**  : The dimension of encoding vectors of the observations.


**Returns**

MLP-based decoder.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L61)
```python
.forward(
   z: Tensor, obs: Tensor
)
```


----


## VAE
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L191)
```python 
VAE(
   device: torch.device, obs_shape: Tuple, latent_dim: int, action_dim: int
)
```


---
Variational auto-encoder for reconstructing transition proces.


**Args**

* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **obs_shape**  : The data shape of observations.
* **latent_dim**  : The dimension of encoding vectors of the observations.
* **action_dim**  : The dimension of predicted actions.



**Methods:**


### .reparameterize
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L222)
```python
.reparameterize(
   mu: Tensor, logvar: Tensor, device: torch.device, training: bool = True
)
```


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xplore/reward/girm.py\#L233)
```python
.forward(
   obs: Tensor, next_obs: Tensor
)
```

