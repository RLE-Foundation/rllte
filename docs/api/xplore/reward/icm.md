#


## ICM
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/reward/icm.py/#L110)
```python 
ICM(
   obs_shape: Tuple, action_shape: Tuple, action_type: str, device: torch.device,
   beta: float, kappa: float, latent_dim: int, lr: float, batch_size: int
)
```


---
Curiosity-Driven Exploration by Self-Supervised Prediction.
See paper: http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf


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


**Returns**

Instance of ICM.


**Methods:**


### .compute_irs
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/reward/icm.py/#L195)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/reward/icm.py/#L256)
```python
.update(
   rollouts: Dict
)
```

---
Update the intrinsic reward module if necessary.


**Args**

* **rollouts**  : The collected experiences. A python dict like
    {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
    actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
    rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.


**Returns**

None

----


## CnnEncoder
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/reward/icm.py/#L11)
```python 
CnnEncoder(
   obs_shape: Tuple, latent_dim: int
)
```


---
Encoder for encoding image-based observations.


**Args**

* **obs_shape**  : The data shape of observations.
* **latent_dim**  : The dimension of encoding vectors of the observations.


**Returns**

CNN-based encoder.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/reward/icm.py/#L50)
```python
.forward(
   obs: Tensor, next_obs: Tensor
)
```


----


## InverseForwardDynamicsModel
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/reward/icm.py/#L64)
```python 
InverseForwardDynamicsModel(
   latent_dim: int, action_dim: int
)
```


---
Inverse-Forward model for reconstructing transition process.


**Args**

* **latent_dim**  : The dimension of encoding vectors of the observations.
* **action_dim**  : The dimension of predicted actions.


**Returns**

Model instance.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/reward/icm.py/#L90)
```python
.forward(
   obs: Tensor, action: Tensor, next_obs: Tensor, training: bool = True
)
```

