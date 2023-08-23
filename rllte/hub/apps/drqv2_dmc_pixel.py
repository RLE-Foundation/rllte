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
import os

from rllte.agent import DrQv2
from rllte.env import make_dmc_env

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="finger_spin")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    # create env
    env = make_dmc_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        from_pixels=True,
        visualize_reward=False,
        frame_stack=3,
        action_repeat=2,
    )
    eval_env = make_dmc_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        from_pixels=True,
        visualize_reward=False,
        frame_stack=3,
        action_repeat=2,
    )
    # create agent
    agent = DrQv2(
        env=env,
        eval_env=eval_env,
        tag=f"drqv2_dmc_pixel_{args.env_id}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        feature_dim=50,
        batch_size=256,
        lr=0.0001,
        eps=1e-8,
        hidden_dim=1024,
        critic_target_tau=0.01,
        update_every_steps=2,
        init_fn="orthogonal",
    )
    # training
    agent.train(num_train_steps=250000, log_interval=500, th_compile=False)
