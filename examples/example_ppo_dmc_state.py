import os
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_dmc_env
from hsuanwu.common.engine import HsuanwuEngine

train_env = make_dmc_env(env_id='cartpole_swingup', 
                         num_envs=1,
                         seed=1, 
                         visualize_reward=True,
                         from_pixels=False,
                         device='cpu'
                         )

test_env = make_dmc_env(env_id='cartpole_swingup',
                        num_envs=1,
                        seed=1, 
                        visualize_reward=True,
                        from_pixels=False,
                        device='cpu'
                        )

print(train_env.observation_space)

@hydra.main(version_base=None, config_path='../cfgs/task', config_name='ppo_dmc_state_config')
def main(cfgs):
    engine = HsuanwuEngine(cfgs=cfgs, train_env=train_env, test_env=test_env)
    engine.invoke()

if __name__ == '__main__':
    main()