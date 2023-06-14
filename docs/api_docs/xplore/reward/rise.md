#


## RISE
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rise.py/#L85)
```python 
RISE(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   beta: float = 0.05, kappa: float = 2.5e-05, latent_dim: int = 128, alpha: float = 0.5,
   k: int = 5, average_entropy: bool = False
)
```


---
Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning (RISE).
See paper: https://ieeexplore.ieee.org/abstract/document/9802917/


**Args**

* **observation_space** (Space) : The observation space of environment.
* **action_space** (Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **beta** (float) : The initial weighting coefficient of the intrinsic rewards.
* **kappa** (float) : The decay rate.
* **latent_dim** (int) : The dimension of encoding vectors.
* **alpha** (alpha) : The The order of Rényi entropy.
* **k** (int) : Use the k-th neighbors.
* **average_entropy** (bool) : Use the average of entropy estimation.


**Returns**

Instance of RISE.


**Methods:**


### .compute_irs
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rise.py/#L131)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rise.py/#L167)
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
