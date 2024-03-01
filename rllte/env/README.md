Integrating RL environments in RLLTE is extremely easy! 

Menu
1. Installation
2. Usage

### Installation

Assuming you are running inside a conda environment. 

Atari (add link here)
pip install ale-py==0.8.1

Craftax (add link here)

You will need a Jax GPU-enabled conda environment:

conda create -n rllte jaxlib==*cuda jax python=3.11 -c conda-forge
pip install craftax
pip install brax
pip install -e .[envs]
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 

DMC (add link here)
pip install dm-control

SuperMarioBros (add link here)
pip install gym-super-mario-bros==7.4.0

Minigrid (add link here)
pip install minigrid

Miniworld (add link here)
pip install miniworld

Procgen (add link here)
pip install procgen

Envpool (add link here)
pip install envpool

### Usage

Each environment has a make_env() function in rllte/env/<your_RL_env>/__init__.py and its necessary wrappers in rllte/env/<your_RL_env>/wrappers.py
To add your custom environments, simply follow the same logic as the currently available environments and the RL training will work flawlessly! 

### Example training
from rllte.agent import PPO

# define the agent
agent = PPO()
        

# start training
agent.train(
    num_train_steps=10_000_000,
)
