import os
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.common.engine import Engine

# from hsuanwu.common.engine import utils
# def create_env():
#     return utils.wrap_pytorch(
#         utils.wrap_deepmind(
#             utils.make_atari('PongNoFrameskip-v4'),
#             clip_rewards=False,
#             frame_stack=True,
#             scale=False,
#         )
#     )

# class TrainEnvs:
#     def __init__(self) -> None:
#         self.envs = list()
#         self.num_envs = 45
#         for idx in range(45):
#             gym_env = create_env()
#             if idx == 0:
#                 self.observation_space = gym_env.observation_space
#                 self.action_space = gym_env.action_space
#             seed = idx ^ int.from_bytes(os.urandom(4), byteorder="little")
#             gym_env.seed(seed)
#             env = utils.Environment(gym_env)
#             self.envs.append(env)

# train_env = TrainEnvs()
# obs = train_env.envs[0].initial()['frame']
# print(obs.size())
# quit(0)

from hsuanwu.env import make_atari_env

train_env = make_atari_env(
    env_id='PongNoFrameskip-v4',
    num_envs=45,
    seed=1,
    frame_stack=4,
    device='cuda'
)

@hydra.main(version_base=None, config_path='../cfgs/task', config_name='impala_atari_config')
def main(cfgs):
    engine = Engine(
        cfgs, train_env
    )
    engine.invoke()

if __name__ == "__main__":
    main()