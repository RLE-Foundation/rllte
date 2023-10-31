#


## PPG
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/ppg.py/#L40)
```python 
PPG(
   env: VecEnv, eval_env: Optional[VecEnv] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_steps: int = 128,
   feature_dim: int = 512, batch_size: int = 256, lr: float = 0.00025, eps: float = 1e-05,
   hidden_dim: int = 512, clip_range: float = 0.2, clip_range_vf: float = 0.2,
   vf_coef: float = 0.5, ent_coef: float = 0.01, max_grad_norm: float = 0.5,
   policy_epochs: int = 32, aux_epochs: int = 6, kl_coef: float = 1.0,
   num_aux_mini_batch: int = 4, num_aux_grad_accum: int = 1, discount: float = 0.999,
   init_fn: str = 'xavier_uniform'
)
```


---
Phasic Policy Gradient (PPG).
Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppg_procgen.py


**Args**

* **env** (VecEnv) : Vectorized environments for training.
* **eval_env** (VecEnv) : Vectorized environments for evaluation.
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
* **vf_coef** (float) : Weighting coefficient of value loss.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **max_grad_norm** (float) : Maximum norm of gradients.
* **policy_epochs** (int) : Number of iterations in the policy phase.
* **aux_epochs** (int) : Number of iterations in the auxiliary phase.
* **kl_coef** (float) : Weighting coefficient of divergence loss.
* **num_aux_grad_accum** (int) : Number of gradient accumulation for auxiliary phase update.
* **discount** (float) : Discount factor.
* **init_fn** (str) : Parameters initialization method.

num_aux_mini_batch (int) Number of mini-batches in auxiliary phase.


**Returns**

PPG agent instance.


**Methods:**


### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/ppg.py/#L186)
```python
.update()
```

---
Update function that returns training metrics such as policy loss, value loss, etc..
