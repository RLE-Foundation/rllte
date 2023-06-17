#


## BaseStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L8)
```python 
BaseStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu'
)
```


---
Base class of storage module.


**Args**

* **observation_space** (gym.Space) : The observation space of environment.
* **action_space** (gym.Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.


**Returns**

Instance of the base storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L53)
```python
.add(
   *args
)
```

---
Add samples to the storage.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L57)
```python
.sample(
   *args
)
```

---
Sample from the storage.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L61)
```python
.update(
   *args
)
```

---
Update the storage if necessary.
