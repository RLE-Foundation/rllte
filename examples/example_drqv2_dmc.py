import os
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_dmc_env
from hsuanwu.common.engine import OffPolicyTrainer

train_env = make_dmc_env(env_id='cartpole_balance', 
                       resource_files=None, 
                       img_source=None,
                       total_frames=None,
                       seed=1, 
                       visualize_reward=False, 
                       from_pixels=True, 
                       frame_skip=2, frame_stack=3)

test_env = make_dmc_env(env_id='cartpole_balance',
                       resource_files=None, 
                       img_source=None,
                       total_frames=None,
                       seed=1, 
                       visualize_reward=False, 
                       from_pixels=True, 
                       frame_skip=2, frame_stack=3)

@hydra.main(version_base=None, config_path='../cfgs', config_name='continuous_task_config')
def main(cfgs):
    trainer = OffPolicyTrainer(train_env=train_env, test_env=test_env, cfgs=cfgs)
    trainer.train()

if __name__ == '__main__':
    main()