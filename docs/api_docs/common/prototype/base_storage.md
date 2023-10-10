#


## BaseStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_storage.py/#L36)
```python 
BaseStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str,
   storage_size: int, batch_size: int, num_envs: int
)
```


---
Base class of the storage module.


**Args**

* **observation_space** (gym.Space) : The observation space of environment.
* **action_space** (gym.Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **storage_size** (int) : The size of the storage.
* **batch_size** (int) : Batch size of samples.
* **num_envs** (int) : The number of parallel environments.


**Returns**

Instance of the base storage.


**Methods:**


### .to_torch
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_storage.py/#L74)
```python
.to_torch(
   x: np.ndarray
)
```

---
Convert numpy array to torch tensor.


**Args**

* **x** (np.ndarray) : Numpy array.


**Returns**

Torch tensor.

### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_storage.py/#L86)
```python
.reset()
```

---
Reset the storage.

### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_storage.py/#L92)
```python
.add(
   *args, **kwargs
)
```

---
Add samples to the storage.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_storage.py/#L96)
```python
.sample(
   *args, **kwargs
)
```

---
Sample from the storage.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/prototype/base_storage.py/#L100)
```python
.update(
   *args, **kwargs
)
```

---
Update the storage if necessary.
