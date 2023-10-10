#


## RISE
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rise.py/#L87)
```python 
RISE(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   beta: float = 0.05, kappa: float = 2.5e-05, latent_dim: int = 128,
   storage_size: int = 10000, num_envs: int = 1, alpha: float = 0.5, k: int = 5,
   average_entropy: bool = False
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
* **storage_size** (int) : The size of the storage for random embeddings.
* **num_envs** (int) : The number of parallel environments.
* **alpha** (alpha) : The The order of Rényi entropy.
* **k** (int) : Use the k-th neighbors.
* **average_entropy** (bool) : Use the average of entropy estimation.


**Returns**

Instance of RISE.


**Methods:**


### .compute_irs
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rise.py/#L142)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rise.py/#L180)
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

### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/rise.py/#L194)
```python
.add(
   samples: Dict
)
```

---
Calculate the random embeddings and insert them into the storage.


**Args**

* **samples**  : The collected samples. A python dict like
    {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
    actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
    rewards (n_steps, n_envs) <class 'th.Tensor'>,
    next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.


**Returns**

None
