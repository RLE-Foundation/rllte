#


## PPO
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/ppo.py\#L16)
```python 
PPO(
   observation_space: Union[gym.Space, DictConfig], action_space: Union[gym.Space,
   DictConfig], device: str, feature_dim: int, lr: float = 0.00025, eps: float = 1e-05,
   hidden_dim: int = 512, clip_range: float = 0.1, clip_range_vf: float = 0.1,
   n_epochs: int = 4, vf_coef: float = 0.5, ent_coef: float = 0.01, aug_coef: float = 0.1,
   max_grad_norm: float = 0.5, network_init_method: str = 'orthogonal'
)
```


---
Proximal Policy Optimization (PPO) agent.
When 'augmentation' module is invoked, this learner will transform into Data Regularized Actor-Critic (DrAC) agent.
Based on: https://github.com/yuanmingqi/pytorch-a2c-ppo-acktr-gail


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **action_space** (Space or DictConfig) : The action space of environment. When invoked by Hydra,
    'action_space' is a 'DictConfig' like
    {"shape": action_space.shape, "n": action_space.n, "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim** (int) : Number of features extracted by the encoder.
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

PPO learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/ppo.py\#L85)
```python
.train(
   training: bool = True
)
```

---
Set the train mode.


**Args**

* **training** (bool) : True (training) or False (testing).


**Returns**

None.

### .integrate
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/ppo.py\#L97)
```python
.integrate(
   **kwargs
)
```

---
Integrate agent and other modules (encoder, reward, ...) together

### .get_value
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/ppo.py\#L117)
```python
.get_value(
   obs: th.Tensor
)
```

---
Get estimated values for observations.


**Args**

* **obs** (Tensor) : Observations.


**Returns**

Estimated values.

### .act
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/ppo.py\#L128)
```python
.act(
   obs: th.Tensor, training: bool = True, step: int = 0
)
```

---
Sample actions based on observations.


**Args**

* **obs**  : Observations.
* **training**  : training mode, True or False.
* **step**  : Global training step.


**Returns**

Sampled actions.

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/ppo.py\#L146)
```python
.update(
   rollout_storage: Storage, episode: int = 0
)
```

---
Update the learner.


**Args**

* **rollout_storage** (Storage) : Hsuanwu rollout storage.
* **episode** (int) : Global training episode.


**Returns**

Training metrics such as actor loss, critic_loss, etc.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/ppo.py\#L239)
```python
.save(
   path: Path
)
```

---
Save models.


**Args**

* **path** (Path) : Storage path.


**Returns**

None.

### .load
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/ppo.py\#L254)
```python
.load(
   path: str
)
```

---
Load initial parameters.


**Args**

* **path** (str) : Import path.


**Returns**

None.
