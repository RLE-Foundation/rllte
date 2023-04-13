import os
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.common.engine import Engine

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