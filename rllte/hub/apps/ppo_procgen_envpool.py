# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


import argparse
import torch as th
th.set_float32_matmul_precision('high')

from rllte.agent import PPO
from rllte.env import make_envpool_procgen_env
from rllte.xploit.encoder import EspeholtResidualEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="bigfish")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    # create env
    env = make_envpool_procgen_env(
        env_id=args.env_id,
        device=args.device,
        num_envs=64,
        seed=args.seed,
        gamma=0.99,
        num_levels=200,
        start_level=0,
        distribution_mode="easy",
    )
    eval_env = make_envpool_procgen_env(
        env_id=args.env_id,
        device=args.device,
        num_envs=1,
        seed=args.seed,
        gamma=0.99,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
        parallel=False
    )
    # create agent
    feature_dim = 256
    agent = PPO(
        env=env,
        eval_env=eval_env,
        tag=f"ppo_procgen_{args.env_id}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        num_steps=256,
        feature_dim=feature_dim,
        batch_size=2048,
        lr=5e-4,
        eps=1e-5,
        clip_range=0.2,
        clip_range_vf=0.2,
        n_epochs=3,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        init_fn="xavier_uniform",
    )
    encoder = EspeholtResidualEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
    agent.set(encoder=encoder)
    # training
    agent.train(num_train_steps=25000000, eval_interval=10)
