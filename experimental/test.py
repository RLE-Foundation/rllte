import sys
sys.path.append('../')
from rllte.env import make_atari_env
import torch as th
from rnd import RND
from icm import ICM
from ride import RIDE
from e3b import E3B
from IPython import embed

device = th.device("cuda:0")
envs = make_atari_env(device=device)
rnd = E3B(
    observation_space=envs.observation_space,
    action_space=envs.action_space,
    device=device, n_envs=8
)

obs, infos = envs.reset()
for i in range(10):
    for j in range(128):
        actions = th.randint(0, 2, (8,))
        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        # watch
        rnd.watch(obs, actions, rewards, terms, truncs, next_obs)
        obs = next_obs

obs = th.randn(128, 8, 4, 84, 84)
actions_tensor = th.randint(0, 2, (128, 8))
samples = {
    "observations": obs,
    "actions": actions_tensor,
    "next_observations": obs,
    "terminateds": th.zeros_like(actions_tensor),
    "truncateds": th.zeros_like(actions_tensor),
}
intrinsic_rewards = rnd.compute(samples)
rnd.update(samples)