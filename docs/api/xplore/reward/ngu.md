#


## NGU
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ngu.py/#L67)
```python 
NGU(
   obs_shape: Tuple, action_shape: Tuple, action_type: str, device: torch.device,
   beta: float, kappa: float, latent_dim: int, lr: float, batch_size: int
)
```


---
Never Give Up: Learning Directed Exploration Strategies (NGU).
See paper: https://arxiv.org/pdf/2002.06038


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

Instance of NGU.


**Methods:**


### .compute_irs
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ngu.py/#L117)
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
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ngu.py/#L164)
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

### .pseudo_counts
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ngu.py/#L197)
```python
.pseudo_counts(
   encoded_obs, k = 10, kernel_cluster_distance = 0.008, kernel_epsilon = 0.0001,
   c = 0.001, sm = 8
)
```


----


## CnnEncoder
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ngu.py/#L12)
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
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ngu.py/#L37)
```python
.forward(
   obs: Tensor
)
```


----


## MlpEncoder
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ngu.py/#L45)
```python 
MlpEncoder(
   obs_shape: Tuple, latent_dim: int
)
```


---
Encoder for encoding state-based observations.


**Args**

* **obs_shape**  : The data shape of observations.
* **latent_dim**  : The dimension of encoding vectors of the observations.


**Returns**

MLP-based encoder.


**Methods:**


### .forward
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ngu.py/#L62)
```python
.forward(
   obs: Tensor
)
```

