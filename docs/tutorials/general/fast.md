# Fast Algorithm Development

Developers only need three steps to implement an RL algorithm with **RLLTE**:

!!! abstract "Workflow"
    1. Selection an algorithm prototype;
    2. Select desired modules;
    3. Write a update function.

The following example illustrates how to write an Advantage Actor-Critic (A2C) agent to solve Atari games.

## Set prototype
Firstly, we select `OnPolicyAgent` as the prototype
``` py
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

import torch as th

class A2C(OnPolicyAgent):
    def __init__(self, env, tag, device, num_steps):
        super().__init__(env=env, tag=tag, device=device, num_steps=num_steps)
        # add some hyper parameters
        lr = 2.5e-4
        batch_size = 256
        num_envs = 8
        # add encoder
        encoder = MnihCnnEncoder(observation_space=env.observation_space, feature_dim=512)
        # add policy
        policy = OnPolicySharedActorCritic(observation_space=env.observation_space,
                                           action_space=env.action_space,
                                           feature_dim=512,
                                           hidden_dim=512,
                                           opt_class=th.optim.Adam,
                                           opt_kwargs=dict(lr=lr),
                                           init_fn="orthogonal")
        # add storage
        storage = VanillaRolloutStorage(observation_space=env.observation_space,
                                        action_space=env.action_space,
                                        device=device,
                                        num_steps=num_steps,
                                        num_envs=num_envs,
                                        batch_size=batch_size)
        # set all the modules [essential operation!!!]
        self.set(encoder=encoder, policy=policy, storage=storage, distribution=Categorical)
```

## Set update function
Finally, we need to fill in the `.update` function:
``` py
def update(self):
    n_epochs = 4
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5

    for _ in range(n_epochs):
        for batch in self.storage.sample():
            # evaluate sampled actions
            new_values, new_log_probs, entropy = self.policy.evaluate_actions(obs=batch.observations, actions=batch.actions)
            # policy loss part
            policy_loss = -(batch.adv_targ * new_log_probs).mean()
            # value loss part
            value_loss = 0.5 * (new_values.flatten() - batch.returns).pow(2).mean()
            # update
            self.policy.opt.zero_grad(set_to_none=True)
            loss = value_loss * vf_coef + policy_loss - entropy * ent_coef
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
            self.policy.opt.step()
```

## Start training
Now we can start training by
``` py title="train.py"
if __name__ == "__main__":
    device = "cuda"
    env = make_atari_env(device=device, num_envs=8)
    agent = A2C(env=env, tag="a2c", device=device, num_steps=128)
    agent.train(num_train_steps=10000)
```
Run `train.py` and you will see the following output:
``` sh
[08/04/2023 02:19:06 PM] - [INFO.] - Invoking RLLTE Engine...
[08/04/2023 02:19:06 PM] - [INFO.] - ================================================================================
[08/04/2023 02:19:06 PM] - [INFO.] - Tag               : a2c
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

The complete code is as follows
``` py
from rllte.common import OnPolicyAgent
from rllte.xploit.encoder import MnihCnnEncoder
from rllte.xploit.policy import OnPolicySharedActorCritic
from rllte.xploit.storage import VanillaRolloutStorage
from rllte.xplore.distribution import Categorical
from rllte.env import make_atari_env

from torch import nn
import torch as th

class A2C(OnPolicyAgent):
    def __init__(self, env, tag, device, num_steps):
        super().__init__(env=env, tag=tag, device=device, num_steps=num_steps)
        # add some hyper parameters
        lr = 2.5e-4
        batch_size = 256
        num_envs = 8
        # add encoder
        encoder = MnihCnnEncoder(observation_space=env.observation_space, feature_dim=512)
        # add policy
        policy = OnPolicySharedActorCritic(observation_space=env.observation_space,
                                           action_space=env.action_space,
                                           feature_dim=512,
                                           hidden_dim=512,
                                           opt_class=th.optim.Adam,
                                           opt_kwargs=dict(lr=lr),
                                           init_fn="orthogonal")
        # add storage
        storage = VanillaRolloutStorage(observation_space=env.observation_space,
                                        action_space=env.action_space,
                                        device=device,
                                        num_steps=num_steps,
                                        num_envs=num_envs,
                                        batch_size=batch_size)
        # set all the modules [essential operation!!!]
        self.set(encoder=encoder, policy=policy, storage=storage, distribution=Categorical)
    
    def update(self):
        n_epochs = 4
        vf_coef = 0.5
        ent_coef = 0.01
        max_grad_norm = 0.5

        for _ in range(n_epochs):
            for batch in self.storage.sample():
                # evaluate sampled actions
                new_values, new_log_probs, entropy = self.policy.evaluate_actions(obs=batch.observations, actions=batch.actions)
                # policy loss part
                policy_loss = -(batch.adv_targ * new_log_probs).mean()
                # value loss part
                value_loss = 0.5 * (new_values.flatten() - batch.returns).pow(2).mean()
                # update
                self.policy.opt.zero_grad(set_to_none=True)
                loss = value_loss * vf_coef + policy_loss - entropy * ent_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.policy.opt.step()

if __name__ == "__main__":
    device = "cuda"
    env = make_atari_env(device=device, num_envs=8)
    agent = A2C(env=env, tag="a2c", device=device, num_steps=128)
    agent.train(num_train_steps=10000)
```