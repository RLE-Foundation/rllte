#


## RE3
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/re3.py/#L64)
```python 
RE3(
   obs_shape: Tuple, action_shape: Tuple, action_type: str, device: torch.device,
   beta: float, kappa: float, latent_dim: int
)
```


---
State Entropy Maximization with Random Encoders for Efficient Exploration (RE3). 
See paper: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf


**Args**

* **obs_shape**  : Data shape of observation.
* **action_space**  : Data shape of action.
* **action_type**  : Continuous or discrete action. "cont" or "dis".
* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **beta**  : The initial weighting coefficient of the intrinsic rewards.
* **kappa**  : The decay rate.
* **latent_dim**  : The dimension of encoding vectors of the observations.


**Returns**

Instance of RE3.


**Methods:**


### .compute_irs
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/re3.py/#L103)
```python
.compute_irs(
   rollouts: Dict, step: int, k: int = 3, average_entropy: bool = False
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
* **k**  : The k value for marking neighbors.
* **average_entropy**  : Use the average of entropy estimation.


**Returns**

The intrinsic rewards

----


## RandomMlpEncoder
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/re3.py/#L43)
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
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/re3.py/#L60)
```python
.forward(
   obs: Tensor
)
```


----


## RandomCnnEncoder
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/re3.py/#L10)
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
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/re3.py/#L35)
```python
.forward(
   obs: Tensor
)
```

