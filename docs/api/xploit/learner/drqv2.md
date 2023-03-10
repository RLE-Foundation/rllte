#


## Actor
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L8)
```python 
Actor(
   action_space: Space, feature_dim: int = 64, hidden_dim: int = 1024
)
```


---
Actor network


**Args**

* **action_space**  : Action space of the environment.
* **features_dim**  : Number of features accepted.
* **hidden_dim**  : Number of units per hidden layer.


**Returns**

Actor network.


**Methods:**


### .forward
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L33)
```python
.forward(
   obs: Tensor
)
```


----


## Critic
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L42)
```python 
Critic(
   action_space: Space, feature_dim: int = 64, hidden_dim: int = 1024
)
```


---
Critic network


**Args**

* **action_space**  : Action space of the environment.
* **features_dim**  : Number of features accepted.
* **hidden_dim**  : Number of units per hidden layer.


**Returns**

Critic network.


**Methods:**


### .forward
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L72)
```python
.forward(
   obs: Tensor, action: Tensor
)
```


----


## DrQv2Agent
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L82)
```python 
DrQv2Agent(
   observation_space: Space, action_space: Space, device: torch.device = 'cuda',
   feature_dim: int = 50, hidden_dim: int = 1024, lr: float = 0.0001,
   critic_target_tau: float = 0.01, num_expl_steps: int = 2000,
   update_every_steps: int = 2, stddev_schedule: str = 'linear(1.0, 0.1, 100000)',
   stddev_clip: float = 0.3
)
```


---
Learner for continuous control tasks.
Current learner: DrQ-v2
Paper: Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning
Link: https://openreview.net/pdf?id=_SJ-_yyes8


**Args**

* **obs_space**  : The observation shape of the environment.
* **action_shape**  : The action shape of the environment.
* **feature_dim**  : Number of features extracted.
* **hidden_dim**  : The size of the hidden layers.
* **lr**  : The learning rate.
* **critic_target_tau**  : The critic Q-function soft-update rate.
* **update_every_steps**  : The agent update frequency.
* **num_expl_steps**  : The exploration steps.
* **stddev_schedule**  : The exploration std schedule.
* **stddev_clip**  : The exploration std clip range.


**Returns**

Agent instance.


**Methods:**


### .train
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L149)
```python
.train(
   training = True
)
```


### .set_encoder
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L156)
```python
.set_encoder(
   encoder
)
```


### .set_dist
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L161)
```python
.set_dist(
   dist
)
```


### .set_aug
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L165)
```python
.set_aug(
   aug
)
```


### .set_irs
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L169)
```python
.set_irs(
   irs
)
```


### .act
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L173)
```python
.act(
   obs: ndarray, training: bool = True, step: int = 0
)
```


### .update
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L192)
```python
.update(
   replay_iter: DataLoader, step: int = 0
)
```


### .update_critic
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L225)
```python
.update_critic(
   obs: Tensor, action: Tensor, reward: Tensor, discount: Tensor, next_obs,
   step: int
)
```


### .update_actor
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L252)
```python
.update_actor(
   obs: Tensor, step: int
)
```

