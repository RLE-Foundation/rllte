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

from rllte.agent import A2C
from rllte.env import make_atari_env

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="SpaceInvadersNoFrameskip-v4")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    # create env
    env = make_atari_env(
        env_id=args.env_id,
        num_envs=8,
        device=args.device,
        seed=args.seed,
    )
    eval_env = make_atari_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
    )
    # create agent
    feature_dim = 512
    agent = A2C(
        env=env,
        eval_env=eval_env,
        tag=f"a2c_atari_{args.env_id}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        num_steps=128,
        feature_dim=feature_dim,
        batch_size=256,
        lr=2.5e-4,
        eps=1e-5,
        n_epochs=4,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        init_fn="orthogonal",
    )
    # training
    agent.train(num_train_steps=50000000)
