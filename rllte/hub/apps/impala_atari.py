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
@inproceedings{espeholt2018impala,
  title={Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures},
  author={Espeholt, Lasse and Soyer, Hubert and Munos, Remi and Simonyan, Karen and Mnih, Vlad and Ward, 
    Tom and Doron, Yotam and Firoiu, Vlad and Harley, Tim and Dunning, Iain and others},
  booktitle={International conference on machine learning},
  pages={1407--1416},
  year={2018},
  organization={PMLR}
}
"""

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"

from rllte.agent import IMPALA  # noqa: E402
from rllte.env import make_atari_env  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-train-steps", type=int, default=30000000)

if __name__ == "__main__":
    args = parser.parse_args()
    # create env
    env = make_atari_env(env_id=args.env_id, device=args.device, seed=args.seed, num_envs=45, parallel=False)

    eval_env = make_atari_env(env_id=args.env_id, device=args.device, seed=args.seed, num_envs=1, parallel=False)

    # create agent
    agent = IMPALA(
        env=env,
        eval_env=eval_env,
        tag=f"impala_atari_{args.env_id}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        num_steps=80,
        num_actors=45,
        num_learners=4,
        num_storages=60,
        feature_dim=512,
    )
    # training
    agent.train(num_train_steps=args.num_train_steps)
