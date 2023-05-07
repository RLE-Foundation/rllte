#


## DistributedStorage
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/distributed_storage.py/#L11)
```python 
DistributedStorage(
   observation_space: Union[gym.Space, DictConfig], action_space: Union[gym.Space,
   DictConfig], device: str = 'cpu', num_steps: int = 100, num_storages: int = 80,
   batch_size: int = 32
)
```


---
Distributed storage for distributed algorithms like IMPALA.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **action_space** (Space or DictConfig) : The action space of environment. When invoked by Hydra,
    'action_space' is a 'DictConfig' like
    {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **num_steps** (int) : The sample steps of per rollout.
* **num_storages** (int) : The number of shared-memory storages.
* **batch_size** (int) : The batch size.


**Returns**

Vanilla rollout storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/distributed_storage.py/#L85)
```python
.add(
   *args
)
```

---
Add sampled transitions into storage.

### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/distributed_storage.py/#L89)
```python
.sample(
   device: th.device, batch_size: int, free_queue: th.multiprocessing.SimpleQueue,
   full_queue: th.multiprocessing.SimpleQueue, storages: List,
   init_actor_state_storages: List, lock = threading.Lock()
)
```

---
Sample transitions from the storage.


**Args**

* **device** (Device) : Device (cpu, cuda, ...) on which the code should be run.
* **batch_size** (int) : The batch size.
* **free_queue** (Queue) : Free queue for communication.
* **full_queue** (Queue) : Full queue for communication.
* **storages** (List[Storage]) : A list of shared storages.
* **init_actor_state_storages**  : (List[Tensor]): Initial states for LSTM.
* **lock** (Lock) : Thread lock.


**Returns**

Batched samples.

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/distributed_storage.py/#L124)
```python
.update(
   *args
)
```

---
Update the storage
