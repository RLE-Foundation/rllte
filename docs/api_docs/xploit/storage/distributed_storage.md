#


## DistributedStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/distributed_storage.py/#L34)
```python 
DistributedStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   num_steps: int = 100, num_storages: int = 80, batch_size: int = 32
)
```


---
Distributed storage for distributed algorithms like IMPALA.


**Args**

* **observation_space** (gym.Space) : The observation space of environment.
* **action_space** (gym.Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **num_steps** (int) : The sample steps of per rollout.
* **num_storages** (int) : The number of shared-memory storages.
* **batch_size** (int) : The batch size.


**Returns**

Vanilla rollout storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/distributed_storage.py/#L98)
```python
.add(
   *args
)
```

---
Add sampled transitions into storage.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/distributed_storage.py/#L101)
```python
.sample(
   free_queue: th.multiprocessing.SimpleQueue,
   full_queue: th.multiprocessing.SimpleQueue, init_actor_state_storages: List,
   lock = threading.Lock()
)
```

---
Sample transitions from the storage.


**Args**

* **free_queue** (Queue) : Free queue for communication.
* **full_queue** (Queue) : Full queue for communication.
* **init_actor_state_storages**  : (List[th.Tensor]): Initial states for LSTM.
* **lock** (Lock) : Thread lock.


**Returns**

Batched samples.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/distributed_storage.py/#L131)
```python
.update(
   *args
)
```

---
Update the storage
