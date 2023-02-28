#


## BaseIntrinsicRewardModule
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/base.py/#L3)
```python 
BaseIntrinsicRewardModule(
   env: Env, device: torch.device, beta: float, kappa: float
)
```


---
Base class of intrinsic reward module.


**Args**

* **env**  : The environment.
* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **beta**  : The initial weighting coefficient of the intrinsic rewards.
* **kappa**  : The decay rate.


**Returns**

The base class of intrinsic reward module.


**Methods:**


### .update
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/base.py/#L52)
```python
.update(
   rollouts: Dict
)
```

---
Update the intrinsic reward module if necessary.


**Args**

* **rollouts**  : {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}


**Returns**

None

### .compute_irs
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/base.py/#L37)
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
    rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}
* **step**  : The current time step.


**Returns**

The intrinsic rewards.
