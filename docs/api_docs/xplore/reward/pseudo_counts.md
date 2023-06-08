#


## PseudoCounts
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/pseudo_counts.py/#L82)
```python 
PseudoCounts(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   beta: float = 0.05, kappa: float = 2.5e-05, latent_dim: int = 32, lr: float = 0.001,
   batch_size: int = 64, capacity: int = 1000, k: int = 10,
   kernel_cluster_distance: float = 0.008, kernel_epsilon: float = 0.0001,
   c: float = 0.001, sm: float = 8.0
)
```


---
Pseudo-counts based on "Never Give Up: Learning Directed Exploration Strategies (NGU)".
See paper: https://arxiv.org/pdf/2002.06038


**Args**

* **observation_space** (Space) : The observation space of environment.
* **action_space** (Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **beta** (float) : The initial weighting coefficient of the intrinsic rewards.
* **kappa** (float) : The decay rate.
* **latent_dim** (int) : The dimension of encoding vectors.
* **lr** (float) : The learning rate.
* **batch_size** (int) : The batch size for update.
* **capacity** (int) : The of capacity the episodic memory.
* **k** (int) : Number of neighbors.
* **kernel_cluster_distance** (float) : The kernel cluster distance.
* **kernel_epsilon** (float) : The kernel constant.
* **c** (float) : The pseudo-counts constant.
* **sm** (float) : The kernel maximum similarity.


**Returns**

Instance of PseudoCounts.


**Methods:**


### .pseudo_counts
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/pseudo_counts.py/#L144)
```python
.pseudo_counts(
   e: th.Tensor
)
```

---
Pseudo counts.


**Args**

* **e** (Tensor) : Encoded observations.


**Returns**

Conut values.

### .compute_irs
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/pseudo_counts.py/#L171)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/reward/pseudo_counts.py/#L208)
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
