# Fast Algorithm Development

<div class="badge">
<a href="https://colab.research.google.com/github/RLE-Foundation/rllte/blob/main/examples/fast_algorithm_dev.ipynb">
<img src="../../../assets/images/colab-logo.svg" style="height: 32px; vertical-align:middle;">
Open in Colab
</a>
</div>

<div class="badge">
<a href="https://github.com/RLE-Foundation/rllte/blob/main/examples/fast_algorithm_dev.ipynb">
<img src="../../../assets/images/github-logo.svg" style="height: 32px; vertical-align:middle;">
&nbsp;&nbsp;View on GitHub
</a>
</div>

Developers only need three steps to implement an RL algorithm with **RLLTE**:

!!! abstract "Workflow"
    1. Select an algorithm prototype;
    2. Select desired modules;
    3. Write a update function.

The following example illustrates how to write an Advantage Actor-Critic (A2C) agent to solve Atari games.

## Set prototype
Firstly, we select `OnPolicyAgent` as the prototype
``` py
from rllte.common.prototype import OnPolicyAgent

class A2C(OnPolicyAgent):
    def __init__(self, env, tag, device, num_steps):
        # here we only use four arguments
        super().__init__(env=env, tag=tag, device=device, num_steps=num_steps)
```
## Set necessary modules
Now we need an `encoder` to process observations, a learnable `policy` to generate actions, and a `storage` to store and sample experiences.
``` py
from rllte.xploit.encoder import MnihCnnEncoder
from rllte.xploit.policy import OnPolicySharedActorCritic
from rllte.xploit.storage import VanillaRolloutStorage
from rllte.xplore.distribution import Categorical
```

## Set update function
Run the `.describe` function of the selected policy and you will see the following output:
``` py
OnPolicySharedActorCritic.describe()

# Output:
# ================================================================================
# Name       : OnPolicySharedActorCritic
# Structure  : self.encoder (shared by actor and critic), self.actor, self.critic
# Forward    : obs -> self.encoder -> self.actor -> actions
#            : obs -> self.encoder -> self.critic -> values
#            : actions -> log_probs
# Optimizers : self.optimizers['opt'] -> (self.encoder, self.actor, self.critic)
# ================================================================================
```
This will illustrate the structure of the policy and indicate the optimizable parts. Finally, merge these modules and write a `.update` function:
``` py
from torch import nn
import torch as th

class A2C(OnPolicyAgent):
    def __init__(self, env, tag, seed, device, num_steps) -> None:
        super().__init__(env=env, tag=tag, seed=seed, device=device, num_steps=num_steps)
        # create modules
        encoder = MnihCnnEncoder(observation_space=env.observation_space, feature_dim=512)
        policy = OnPolicySharedActorCritic(observation_space=env.observation_space,
                                           action_space=env.action_space,
                                           feature_dim=512,
                                           opt_class=th.optim.Adam,
                                           opt_kwargs=dict(lr=2.5e-4, eps=1e-5),
                                           init_fn="xavier_uniform"
                                           )
        storage = VanillaRolloutStorage(observation_space=env.observation_space,
                                        action_space=env.action_space,
                                        device=device,
                                        storage_size=self.num_steps,
                                        num_envs=self.num_envs,
                                        batch_size=256
                                        )
        # set all the modules
        self.set(encoder=encoder, policy=policy, storage=storage, distribution=Categorical)
    
    def update(self):
        for _ in range(4):
            for batch in self.storage.sample():
                # evaluate the sampled actions
                new_values, new_log_probs, entropy = self.policy.evaluate_actions(obs=batch.observations, actions=batch.actions)
                # policy loss part
                policy_loss = - (batch.adv_targ * new_log_probs).mean()
                # value loss part
                value_loss = 0.5 * (new_values.flatten() - batch.returns).pow(2).mean()
                # update
                self.policy.optimizers['opt'].zero_grad(set_to_none=True)
                (value_loss * 0.5 + policy_loss - entropy * 0.01).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy.optimizers['opt'].step()
```

## Start training
Now we can start training by
``` py title="train.py"
from rllte.env import make_atari_env
if __name__ == "__main__":
    device = "cuda"
    env = make_atari_env("AlienNoFrameskip-v4", num_envs=8, seed=0, device=device)
    agent = A2C(env=env, tag="a2c_atari", seed=0, device=device, num_steps=128)
    agent.train(num_train_steps=10000000)
```
Run `train.py` and you will see the following output:
``` sh
[08/04/2023 02:19:06 PM] - [INFO.] - Invoking RLLTE Engine...
[08/04/2023 02:19:06 PM] - [INFO.] - ================================================================================
[08/04/2023 02:19:06 PM] - [INFO.] - Tag               : a2c_atari
[08/04/2023 02:19:06 PM] - [INFO.] - Device            : NVIDIA GeForce RTX 3090
[08/04/2023 02:19:07 PM] - [DEBUG] - Agent             : A2C
[08/04/2023 02:19:07 PM] - [DEBUG] - Encoder           : MnihCnnEncoder
[08/04/2023 02:19:07 PM] - [DEBUG] - Policy            : OnPolicySharedActorCritic
[08/04/2023 02:19:07 PM] - [DEBUG] - Storage           : VanillaRolloutStorage
[08/04/2023 02:19:07 PM] - [DEBUG] - Distribution      : Categorical
[08/04/2023 02:19:07 PM] - [DEBUG] - Augmentation      : False
[08/04/2023 02:19:07 PM] - [DEBUG] - Intrinsic Reward  : False
[08/04/2023 02:19:07 PM] - [DEBUG] - ================================================================================
[08/04/2023 02:19:09 PM] - [TRAIN] - S: 1024        | E: 8           | L: 44          | R: 99.000      | FPS: 407.637   | T: 0:00:02    
[08/04/2023 02:19:10 PM] - [TRAIN] - S: 2048        | E: 16          | L: 50          | R: 109.000     | FPS: 594.725   | T: 0:00:03    
[08/04/2023 02:19:11 PM] - [TRAIN] - S: 3072        | E: 24          | L: 47          | R: 96.000      | FPS: 692.433   | T: 0:00:04    
[08/04/2023 02:19:12 PM] - [TRAIN] - S: 4096        | E: 32          | L: 36          | R: 93.000      | FPS: 755.935   | T: 0:00:05    
[08/04/2023 02:19:13 PM] - [TRAIN] - S: 5120        | E: 40          | L: 55          | R: 99.000      | FPS: 809.577   | T: 0:00:06    
[08/04/2023 02:19:14 PM] - [TRAIN] - S: 6144        | E: 48          | L: 46          | R: 34.000      | FPS: 847.310   | T: 0:00:07    
[08/04/2023 02:19:15 PM] - [TRAIN] - S: 7168        | E: 56          | L: 49          | R: 43.000      | FPS: 878.628   | T: 0:00:08   
...
```

As shown in this example, only a few dozen lines of code are needed to create RL agents with **RLLTE**. 