import os
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_procgen_env
from hsuanwu.common.engine import HsuanwuEngine

train_env = make_procgen_env(
    env_id='fruitbot',
    num_envs=64,
    num_levels=200,
    start_level=0,
    distribution_mode='easy',
    device='cuda'
)

test_env = make_procgen_env(
    env_id='fruitbot',
    num_envs=1,
    num_levels=0,
    start_level=0,
    distribution_mode='easy',
    device='cuda'
)

@hydra.main(version_base=None, config_path='../cfgs/task', config_name='ppo_procgen_config')
def main(cfgs):
    engine = HsuanwuEngine(cfgs=cfgs, train_env=train_env, test_env=test_env)
    engine.invoke()

if __name__ == '__main__':
    main()