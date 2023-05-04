#


## DistributedTrainer
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/distributed_trainer.py/#L126)
```python 
DistributedTrainer(
   cfgs: omegaconf.DictConfig, train_env: gym.Env, test_env: gym.Env = None
)
```


---
Trainer for distributed algorithms.


**Args**

* **train_env** (Env) : A list of Gym-like environments for training.
* **test_env** (Env) : A Gym-like environment for testing.
* **cfgs** (DictConfig) : Dict config for configuring RL algorithms.


**Returns**

Distributed trainer instance.


**Methods:**


### .act
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/distributed_trainer.py/#L170)
```python
.act(
   cfgs: omegaconf.DictConfig, logger: Logger, gym_env: gym.Env, actor_idx: int,
   actor_model: nn.Module, free_queue: mp.SimpleQueue, full_queue: mp.SimpleQueue,
   storages: Dict[str, List], init_actor_state_storages: List[th.Tensor]
)
```

---
Sampling function for each actor.


**Args**

* **cfgs** (DictConfig) : Training configs.
* **logger** (Logger) : Hsuanwu logger.
* **gym_env** (Env) : A Gym-like environment.
* **actor_idx** (int) : The index of actor.
* **actor_model** (NNMoudle) : Actor network.
* **free_queue** (Queue) : Free queue for communication.
* **full_queue** (Queue) : Full queue for communication.
* **storages** (List[Storage]) : A list of shared storages.
* **init_actor_state_storages** (List[Tensor]) : Initial states for LSTM.


**Returns**

None.

### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/distributed_trainer.py/#L240)
```python
.train()
```

---
Training function

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/distributed_trainer.py/#L362)
```python
.test()
```

---
Testing function.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/distributed_trainer.py/#L383)
```python
.save()
```

---
Save the trained model.
