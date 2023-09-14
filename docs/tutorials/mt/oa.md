# Observation Augmentation for Sample Efficiency and Generalization

<div class="badge">
<a href="https://colab.research.google.com/github/RLE-Foundation/rllte/blob/main/examples/observation_augmentation.ipynb">
<img src="../../../assets/images/colab-logo.svg" style="height: 32px; vertical-align:middle;">
Open in Colab
</a>
</div>

<div class="badge">
<a href="https://github.com/RLE-Foundation/rllte/blob/main/examples/observation_augmentation.ipynb">
<img src="../../../assets/images/github-logo.svg" style="height: 32px; vertical-align:middle;">
&nbsp;&nbsp;View on GitHub
</a>
</div>

Observation augmentation is an efficient approach to improve sample efficiency and generalization, which is also a basic primitive of **RLLTE**.

> - Laskin M, Lee K, Stooke A, et al. Reinforcement learning with augmented data[J]. Advances in neural information processing systems, 2020, 33: 19884-19895.
> - Yarats D, Fergus R, Lazaric A, et al. Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning[C]//International Conference on Learning Representations. 2021.

**RLLTE** implements the augmentation modules via a PyTorch-NN manner, and both imaged-based and state-based observations are supported. A code example is:
```py title="example.py"
from rllte.agent import DrAC
from rllte.env import make_atari_env
from rllte.xplore.augmentation import RandomCrop

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_atari_env(device=device)
    eval_env = make_atari_env(device=device)
    # create agent
    agent = DrAC(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="drac_atari")
    # create augmentation module
    random_crop = RandomCrop()
    # set the module
    agent.set(augmentation=random_crop)
    # start training
    agent.train(num_train_steps=5000)
```
Run `example.py` and you'll see the augmentation module is invoked:
``` sh
[08/04/2023 05:00:15 PM] - [INFO.] - Invoking RLLTE Engine...
[08/04/2023 05:00:15 PM] - [INFO.] - ================================================================================
[08/04/2023 05:00:15 PM] - [INFO.] - Tag               : drac_atari
[08/04/2023 05:00:16 PM] - [INFO.] - Device            : NVIDIA GeForce RTX 3090
[08/04/2023 05:00:16 PM] - [DEBUG] - Agent             : DrAC
[08/04/2023 05:00:16 PM] - [DEBUG] - Encoder           : MnihCnnEncoder
[08/04/2023 05:00:16 PM] - [DEBUG] - Policy            : OnPolicySharedActorCritic
[08/04/2023 05:00:16 PM] - [DEBUG] - Storage           : VanillaRolloutStorage
[08/04/2023 05:00:16 PM] - [DEBUG] - Distribution      : Categorical
[08/04/2023 05:00:16 PM] - [DEBUG] - Augmentation      : True, RandomCrop
[08/04/2023 05:00:16 PM] - [DEBUG] - Intrinsic Reward  : False
[08/04/2023 05:00:16 PM] - [DEBUG] - ================================================================================
...
```

!!! info "Compatibility of augmentation"
    Note that the module will only make difference when the algorithm supports data augmentation.
    Please refer to [https://docs.rllte.dev/api/](https://docs.rllte.dev/api/) for the compatibility.