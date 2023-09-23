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


"""
The following hyperparameters are from the paper:
@inproceedings{raffin2022smooth,
  title={Smooth exploration for robotic reinforcement learning},
  author={Raffin, Antonin and Kober, Jens and Stulp, Freek},
  booktitle={Conference on Robot Learning},
  pages={1634--1644},
  year={2022},
  organization={PMLR}
}
"""

import argparse

import torch as th

from rllte.agent import PPO
from rllte.env import make_bullet_env

th.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="AntBulletEnv-v0")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-train-steps", type=int, default=2e6)

if __name__ == "__main__":
    args = parser.parse_args()
    # create env
    env = make_bullet_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
    )
    eval_env = make_bullet_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
    )
    # create agent
    feature_dim = 64
    env_name = args.env_id.split("BulletEnv-v0")[0].lower()
    agent = PPO(
        env=env,
        eval_env=eval_env,
        tag=f"ppo_bullet_{env_name}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        pretraining=False,
        num_steps=2048,
        feature_dim=feature_dim,
        batch_size=64,
        lr=2e-4,
        eps=1e-5,
        hidden_dim=512,
        clip_range=0.2,
        clip_range_vf=None,
        n_epochs=10,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        init_fn="orthogonal",
    )
    # training
    agent.train(num_train_steps=args.num_train_steps, eval_interval=10)
