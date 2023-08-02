#


## VanillaDistributedStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_distributed_storage.py/#L35)
```python 
VanillaDistributedStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   num_steps: int = 100, num_storages: int = 80, batch_size: int = 32
)
```


---
Vanilla distributed storage for distributed algorithms like IMPALA.


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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_distributed_storage.py/#L103)
```python
.add(
   idx: int, timestep: int, actor_output: Dict[str, Any], env_output: Dict[str,
   Any]
)
```

---
Add sampled transitions into storage.


**Args**

* **idx** (int) : The index of storage.
* **timestep** (int) : The timestep of rollout.
* **actor_output** (Dict) : Actor output.
* **env_output** (Dict) : Environment output.


**Returns**

None

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_distributed_storage.py/#L125)
```python
.sample(
   free_queue: mp.SimpleQueue, full_queue: mp.SimpleQueue, lock = threading.Lock()
)
```

---
Sample transitions from the storage.


**Args**

* **free_queue** (Queue) : Free queue for communication.
* **full_queue** (Queue) : Full queue for communication.
* **lock** (Lock) : Thread lock.


**Returns**

Batched samples.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_distributed_storage.py/#L151)
```python
.update(
   *args
)
```

---
Update the storage
