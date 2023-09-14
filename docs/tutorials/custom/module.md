# Custom Module

<div class="badge">
<a href="https://colab.research.google.com/github/RLE-Foundation/rllte/blob/main/examples/custom_module.ipynb">
<img src="../../../assets/images/colab-logo.svg" style="height: 32px; vertical-align:middle;">
Open in Colab
</a>
</div>

<div class="badge">
<a href="https://github.com/RLE-Foundation/rllte/blob/main/examples/custom_module.ipynb">
<img src="../../../assets/images/github-logo.svg" style="height: 32px; vertical-align:middle;">
&nbsp;&nbsp;View on GitHub
</a>
</div>

**RLLTE** is an extremely open platform that supports custom modules, including `encoder`, `storage`, `policy`, etc. Just write a new module based on the `BaseClass`, then we can insert it into an agent directly. Suppose we want to build a new encoder entitled `CustomEncoder`. An example is
```py title="example.py"
from rllte.agent import PPO
from rllte.env import make_atari_env
from rllte.common.prototype import BaseEncoder
from gymnasium.spaces import Space
from torch import nn
import torch as th

class CustomEncoder(BaseEncoder):
    """Custom encoder.
    
    Args:
        observation_space (Space): The observation space of environment.
        feature_dim (int): Number of features extracted.

    Returns:
        The new encoder instance.
    """
    def __init__(self, observation_space: Space, feature_dim: int = 0) -> None:
        super().__init__(observation_space, feature_dim)

        obs_shape = observation_space.shape
        assert len(obs_shape) == 3

        self.trunk = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.ones(size=tuple(obs_shape)).float()
            n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

        self.trunk.extend([nn.Linear(n_flatten, feature_dim), nn.ReLU()])

    def forward(self, obs: th.Tensor) -> th.Tensor:
        h = self.trunk(obs / 255.0)

        return h.view(h.size()[0], -1)

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_atari_env(device=device)
    eval_env = make_atari_env(device=device)
    # create agent
    feature_dim = 512
    agent = PPO(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="ppo_atari",
                feature_dim=feature_dim)
    # create a new encoder
    encoder = CustomEncoder(observation_space=env.observation_space, 
                         feature_dim=feature_dim)
    # set the new encoder
    agent.set(encoder=encoder)
    # start training
    agent.train(num_train_steps=5000)
```
Run `example.py` and you'll see the old `MnihCnnEncoder` has been replaced by `CustomEncoder`:
``` sh
[08/04/2023 03:47:24 PM] - [INFO.] - Invoking RLLTE Engine...
[08/04/2023 03:47:24 PM] - [INFO.] - ================================================================================
[08/04/2023 03:47:24 PM] - [INFO.] - Tag               : ppo_atari
[08/04/2023 03:47:24 PM] - [INFO.] - Device            : NVIDIA GeForce RTX 3090
[08/04/2023 03:47:24 PM] - [DEBUG] - Agent             : PPO
[08/04/2023 03:47:24 PM] - [DEBUG] - Encoder           : CustomEncoder
[08/04/2023 03:47:24 PM] - [DEBUG] - Policy            : OnPolicySharedActorCritic
[08/04/2023 03:47:24 PM] - [DEBUG] - Storage           : VanillaRolloutStorage
[08/04/2023 03:47:24 PM] - [DEBUG] - Distribution      : Categorical
[08/04/2023 03:47:24 PM] - [DEBUG] - Augmentation      : False
[08/04/2023 03:47:24 PM] - [DEBUG] - Intrinsic Reward  : False
[08/04/2023 03:47:24 PM] - [DEBUG] - ================================================================================
...
```
As for customizing modules like `Storage` and `Distribution`, etc., users should consider compatibility with specific algorithms.