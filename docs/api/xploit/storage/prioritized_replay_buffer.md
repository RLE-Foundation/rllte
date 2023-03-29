#


## PrioritizedReplayBuffer
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/prioritized_replay_buffer.py/#L5)
```python 
PrioritizedReplayBuffer(
   buffer_size, alpha = 0.6, beta = 0.4, beta_schedule = None, epsilon = 1e-06
)
```




**Methods:**


### .add
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/prioritized_replay_buffer.py/#L17)
```python
.add(
   state, action, reward, next_state, done
)
```


### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/prioritized_replay_buffer.py/#L28)
```python
.sample(
   batch_size
)
```


### .update_priorities
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/prioritized_replay_buffer.py/#L45)
```python
.update_priorities(
   indices, priorities
)
```

