#


## BaseStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L7)
```python 
BaseStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu'
)
```


---
Base class of storage module.


**Args**

* **observation_space** (Space) : The observation space of environment.
* **action_space** (Space) : The action space of environment. 
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.


**Returns**

Instance of the base storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L50)
```python
.add(
   *args
)
```

---
Add sampled transitions into storage.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L54)
```python
.sample(
   *args
)
```

---
Sample from the storage.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L58)
```python
.update(
   *args
)
```

---
Update the storage
