+ **Agent**: *Agent for interacting and learning.*

|Module|Recurrent|Box|Discrete|MultiBinary|Multi Processing|NPU|Paper|Citations|
|:-|:-|:-|:-|:-|:-|:-|:-|:-|
|SAC|❌| ✔️ |❌|❌|❌|🐌 | [Link](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf) |5077⭐|
|DrQ|❌| ✔️ |❌|❌|❌|🐌 | [Link](https://arxiv.org/pdf/2004.13649) |433⭐|
|DDPG|❌| ✔️ |❌|❌|❌|🐌 | [Link](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------) |11819⭐|
|DrQ-v2|❌| ✔️ |❌|❌|❌|🐌 | [Link](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com) |100⭐|
|PPO|❌| ✔️ |✔️|🐌|✔️|🐌 | [Link](https://arxiv.org/pdf/1707.06347) |11155⭐|
|DrAC|❌| ✔️ |✔️|🐌|✔️|🐌 | [Link](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf) |29⭐|
|PPG|❌| ✔️ |✔️|🐌|✔️|🐌| [Link](http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf) |82⭐|
|IMPALA|✔️| ✔️ |✔️|🐌|✔️|🐌| [Link](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf) |1219⭐|

!!! tip "Tips of Agent"
    - 🐌: Developing.
    - **NPU**: Support Neural-network processing unit.
    - **Recurrent**: Support recurrent neural network.
    - **Box**: A N-dimensional box that containes every point in the action space.
    - **Discrete**: A list of possible actions, where each timestep only one of the actions can be used.
    - **MultiBinary**: A list of possible actions, where each timestep any of the actions can be used in any combination.

+ **Encoder**: *Neural nework-based encoder for processing observations.*

|Module|Input|Reference|Target Task|
|:-|:-|:-|:-|
|EspeholtResidualEncoder|Images|[IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)|Atari or Procgen games.|
|IdentityEncoder|States|N/A|DeepMind Control Suite: state|
|MnihCnnEncoder|Images|[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf?source=post_page---------------------------)|Atari games.|
|TassaCnnEncoder|Images|[DeepMind Control Suite](https://arxiv.org/pdf/1801.00690)|DeepMind Control Suite: pixel|
|VanillaMlpEncoder|States|N/A|DeepMind Control Suite: state|

!!! tip "Tips of Encoder"
    - **Naming Rule**: 'Surname of the first author' + 'Backbone' + 'Encoder'
    - **Input**: Input type.
    - **Target Task**: The testing tasks in their paper or potential tasks.

+ **Storage**: *Storage for storing collected experiences.*

|Module|Remark|
|:-|:-|
|VanillaRolloutStorage|On-Policy RL|
|VanillaReplayStorage|Off-Policy RL|
|NStepReplayStorage|Off-Policy RL|
|PrioritizedReplayStorage|Off-Policy RL|
|DistributedStorage|Distributed RL|