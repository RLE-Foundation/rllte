#


## Procgen
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/procgen.py/#L33)
```python 

```


---
Scores and learning cures of various RL algorithms on the full Procgen benchmark.
Environment link: https://github.com/openai/procgen
Number of environments: 16
Number of training steps: 25,000,000
Number of seeds: 10
Added algorithms: [PPO]


**Methods:**


### .load_scores
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/procgen.py/#L45)
```python
.load_scores()
```

---
Returns final performance.

### .load_curves
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/procgen.py/#L59)
```python
.load_curves()
```

---
Returns learning curves using a Dict of arrays:
curves = {
    "eval": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...},
},
    "eval": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...},
},
...
---
}
