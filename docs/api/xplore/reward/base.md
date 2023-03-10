#


## BaseIntrinsicRewardModule
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/base.py/#L5)
```python 
BaseIntrinsicRewardModule(
   obs_shape: Tuple, action_shape: Tuple, action_type: str, device: torch.device,
   beta: float, kappa: float
)
```


---
Base class of intrinsic reward module.


**Args**

* **obs_shape**  : Data shape of observation.
* **action_space**  : Data shape of action.
* **action_type**  : Continuous or discrete action. "cont" or "dis".
* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **beta**  : The initial weighting coefficient of the intrinsic rewards.
* **kappa**  : The decay rate.


**Returns**

Instance of the base intrinsic reward module.


**Methods:**


### .compute_irs
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/base.py/#L46)
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

The intrinsic rewards.

### .update
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xplore/reward/base.py/#L62)
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
