#


## IMPALALearner
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/impala.py\#L184)
```python 
IMPALALearner(
   observation_space: Space, action_space: Space, device: Device, feature_dim: int,
   lr: float, eps: float, use_lstm: bool, ent_coef: float, baseline_coef: float,
   max_grad_norm: float, discount: float
)
```


---
Importance Weighted Actor-Learner Architecture (IMPALA).


**Args**

* **observation_space** (Dict) : Observation space of the environment.
    For supporting Hydra, the original 'observation_space' is transformed into a dict like {"shape": observation_space.shape, }.
* **action_space** (Dict) : Action shape of the environment.
    For supporting Hydra, the original 'action_space' is transformed into a dict like
    {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
* **device** (Device) : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.
* **use_lstm** (bool) : Use LSTM in the policy network or not.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **baseline_coef** (float) : .
* **max_grad_norm** (float) : Maximum norm of gradients.
* **discount** (float) : Discount factor.


**Returns**

IMPALALearner distance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/impala.py\#L237)
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

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/impala.py\#L251)
```python
.update(
   cfgs: DictConfig, actor_model: NNModule, learner_model: NNModule, batch: Batch,
   init_actor_states: Tuple[Tensor, ...], optimizer: torch.optim.Optimizer,
   lr_scheduler: torch.optim.lr_scheduler, lock = threading.Lock()
)
```

---
Update the learner model.


**Args**

* **cfgs** (DictConfig) : Training configs.
* **actor_model** (NNMoudle) : Actor network.
* **learner_model** (NNMoudle) : Learner network.
* **batch** (Batch) : Batch samples.
* **init_actor_states** (List[Tensor]) : Initial states for LSTM.
* **optimizer** (torch.optim.Optimizer) : Optimizer.
* **lr_scheduler** (torch.optim.lr_scheduler) : Learning rate scheduler.
* **lock** (Lock) : Thread lock.


**Returns**

Training metrics.
