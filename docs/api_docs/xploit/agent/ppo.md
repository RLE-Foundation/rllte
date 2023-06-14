#


## PPO
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/ppo.py/#L37)
```python 
PPO(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_steps: int = 128,
   eval_every_episodes: int = 10, feature_dim: int = 512, batch_size: int = 256,
   lr: float = 0.00025, eps: float = 1e-05, hidden_dim: int = 512, clip_range: float = 0.1,
   clip_range_vf: float = 0.1, n_epochs: int = 4, vf_coef: float = 0.5,
   ent_coef: float = 0.01, aug_coef: float = 0.1, max_grad_norm: float = 0.5,
   network_init_method: str = 'orthogonal'
)
```


---
Proximal Policy Optimization (PPO) agent.
When the `augmentation` module is invoked, this agent will transform into Data Regularized Actor-Critic (DrAC) agent.
Based on: https://github.com/yuanmingqi/pytorch-a2c-ppo-acktr-gail


**Args**

* **env** (gym.Env) : A Gym-like environment for training.
* **eval_env** (gym.Env) : A Gym-like environment for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on the pre-training mode.
* **num_steps** (int) : The sample length of per rollout.
* **eval_every_episodes** (int) : Evaluation interval.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **batch_size** (int) : Number of samples per batch to load.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.
* **hidden_dim** (int) : The size of the hidden layers.
* **clip_range** (float) : Clipping parameter.
* **clip_range_vf** (float) : Clipping parameter for the value function.
* **n_epochs** (int) : Times of updating the policy.
* **vf_coef** (float) : Weighting coefficient of value loss.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **aug_coef** (float) : Weighting coefficient of augmentation loss.
* **max_grad_norm** (float) : Maximum norm of gradients.
* **network_init_method** (str) : Network initialization method name.



**Returns**

PPO agent instance.


**Methods:**


### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/ppo.py/#L119)
```python
.freeze()
```

---
Freeze the structure of the agent.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/ppo.py/#L133)
```python
.update()
```

---
Update the agent and return training metrics such as actor loss, critic_loss, etc.
