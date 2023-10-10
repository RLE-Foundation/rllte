#


## Atari
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/atari.py/#L32)
```python 

```


---
Scores and learning cures of various RL algorithms on the full Atari benchmark.
Environment link: https://github.com/Farama-Foundation/Arcade-Learning-Environment
Number of environments: 57
Number of training steps: 50,000,000
Number of seeds: 10
Added algorithms: [PPO]


**Methods:**


### .load_scores
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/atari.py/#L44)
```python
.load_scores()
```

---
Returns final performance.

### .load_curves
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/atari.py/#L55)
```python
.load_curves()
```

---
Returns learning curves using a `Dict` of NumPy arrays:
curves = {
    "eval": {"Pong-v5": np.ndarray(shape=(Number of seeds, Number of points)), ...},
},
...
---
}
