import os
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_dmc_env
from hsuanwu.common.engine import OffPolicyTrainer

train_env = make_dmc_env(env_id='cartpole_swingup', 
                         num_envs=1,
                         seed=1, 
                         visualize_reward=True,
                         from_pixels=False
                         )

test_env = make_dmc_env(env_id='cartpole_swingup',
                        num_envs=1,
                        seed=1, 
                        visualize_reward=True,
                        from_pixels=False
                        )

print(train_env.observation_space)

@hydra.main(version_base=None, config_path='../cfgs/task', config_name='sac_dmc_config')
def main(cfgs):
    trainer = OffPolicyTrainer(cfgs=cfgs, train_env=train_env, test_env=test_env)
    trainer.train()

if __name__ == '__main__':
    main()