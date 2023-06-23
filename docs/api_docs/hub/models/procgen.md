#


## Procgen
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/models/procgen.py/#L31)
```python 

```


---
Trained models various RL algorithms on the full Procgen benchmark.
Environment link: https://github.com/openai/procgen
Number of environments: 16
Number of training steps: 25,000,000
Number of seeds: 10
Added algorithms: [PPO]


**Methods:**


### .load_models
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/models/procgen.py/#L43)
```python
.load_models(
   agent: str, env_id: str, seed: int, device: str = 'cpu'
)
```

---
Load the model from the hub.


**Args**

* **agent** (str) : The agent to load.
* **env_id** (str) : The environment id to load.
* **seed** (int) : The seed to load.
* **device** (str) : The device to load the model on.


**Returns**

The loaded model.
