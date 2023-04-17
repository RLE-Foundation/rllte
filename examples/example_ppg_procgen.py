import os
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_procgen_env
from hsuanwu.common.engine import OnPolicyTrainer

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
print(train_env.action_space)
@hydra.main(version_base=None, config_path='../cfgs/task', config_name='ppg_procgen_config')
def main(cfgs):
    trainer = OnPolicyTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
    trainer.train()

if __name__ == '__main__':
    main()