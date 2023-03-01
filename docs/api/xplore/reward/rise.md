#


## RISE
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/rise.py/#L63)
```python 
RISE(
   env: Env, device: torch.device, beta: float, kappa: float, latent_dim: int
)
```


---
Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning (RISE). 
See paper: https://ieeexplore.ieee.org/abstract/document/9802917/


**Args**

* **env**  : The environment.
* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **beta**  : The initial weighting coefficient of the intrinsic rewards.
* **kappa**  : The decay rate.
* **latent_dim**  : The dimension of encoding vectors of the observations.


**Returns**

Instance of RISE.


**Methods:**


### .compute_irs
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/rise.py/#L98)
```python
.compute_irs(
   rollouts: Dict, step: int, alpha: float = 0.5, k: int = 3,
   average_entropy: bool = False
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
* **alpha**  : The The order of Rényi entropy.
* **k**  : The k value for marking neighbors.
* **average_entropy**  : Use the average of entropy estimation.


**Returns**

The intrinsic rewards

----


## RandomMlpEncoder
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/rise.py/#L42)
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
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/rise.py/#L59)
```python
.forward(
   obs: Tensor
)
```


----


## RandomCnnEncoder
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/rise.py/#L10)
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
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/rise.py/#L34)
```python
.forward(
   obs: Tensor
)
```

