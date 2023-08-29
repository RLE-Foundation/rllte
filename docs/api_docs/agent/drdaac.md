#


## DrDAAC
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/drdaac.py/#L44)
```python 
DrDAAC(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_steps: int = 128,
   feature_dim: int = 512, batch_size: int = 256, lr: float = 0.00025, eps: float = 1e-05,
   hidden_dim: int = 256, clip_range: float = 0.2, clip_range_vf: float = 0.2,
   policy_epochs: int = 1, value_freq: int = 1, value_epochs: int = 9, vf_coef: float = 0.5,
   ent_coef: float = 0.01, aug_coef: float = 0.1, adv_coef: float = 0.25,
   max_grad_norm: float = 0.5, init_fn: str = 'xavier_uniform'
)
```


---
Data-Regularized extension of Decoupled Advantage Actor-Critic (DAAC) agent.
Based on: https://github.com/rraileanu/idaac


**Args**

* **env** (gym.Env) : A Gym-like environment for training.
* **eval_env** (gym.Env) : A Gym-like environment for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on the pre-training mode.
* **num_steps** (int) : The sample length of per rollout.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **batch_size** (int) : Number of samples per batch to load.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.
* **hidden_dim** (int) : The size of the hidden layers.
* **clip_range** (float) : Clipping parameter.
* **clip_range_vf** (float) : Clipping parameter for the value function.
* **policy_epochs** (int) : Times of updating the policy network.
* **value_freq** (int) : Update frequency of the value network.
* **value_epochs** (int) : Times of updating the value network.
* **vf_coef** (float) : Weighting coefficient of value loss.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **aug_coef** (float) : Weighting coefficient of augmentation loss.
* **adv_ceof** (float) : Weighting coefficient of advantage loss.
* **max_grad_norm** (float) : Maximum norm of gradients.
* **init_fn** (str) : Parameters initialization method.



**Returns**

DAAC agent instance.


**Methods:**


### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/drdaac.py/#L178)
```python
.update()
```

---
Update function that returns training metrics such as policy loss, value loss, etc..
