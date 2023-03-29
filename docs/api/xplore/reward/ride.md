#


## RIDE
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ride.py/#L64)
```python 
RIDE(
   obs_shape: Tuple, action_shape: Tuple, action_type: str, device: torch.device,
   beta: float, kappa: float, latent_dim: int
)
```


---
RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments.
See paper: https://arxiv.org/pdf/2002.12292


**Args**

* **obs_shape**  : Data shape of observation.
* **action_space**  : Data shape of action.
* **action_type**  : Continuous or discrete action. "cont" or "dis".
* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **beta**  : The initial weighting coefficient of the intrinsic rewards.
* **kappa**  : The decay rate.
* **latent_dim**  : The dimension of encoding vectors of the observations.


**Returns**

Instance of RIDE.


**Methods:**


### .pseudo_counts
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ride.py/#L103)
```python
.pseudo_counts(
   src_feats, k = 10, kernel_cluster_distance = 0.008, kernel_epsilon = 0.0001,
   c = 0.001, sm = 8
)
```


### .compute_irs
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ride.py/#L128)
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

----


## RandomCnnEncoder
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ride.py/#L10)
```python 
RandomCnnEncoder(
   obs_shape: Tuple, latent_dim: int
)
```


---
Random encoder for encoding image-based observations.


**Args**

* **obs_shape**  : The data shape of observations.
* **latent_dim**  : The dimension of encoding vectors of the observations.


**Returns**

CNN-based random encoder.


**Methods:**


### .forward
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ride.py/#L35)
```python
.forward(
   obs: Tensor
)
```


----


## RandomMlpEncoder
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ride.py/#L43)
```python 
RandomMlpEncoder(
   obs_shape: Tuple, latent_dim: int
)
```


---
Random encoder for encoding state-based observations.


**Args**

* **obs_shape**  : The data shape of observations.
* **latent_dim**  : The dimension of encoding vectors of the observations.


**Returns**

MLP-based random encoder.


**Methods:**


### .forward
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/ride.py/#L60)
```python
.forward(
   obs: Tensor
)
```

