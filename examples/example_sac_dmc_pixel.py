import os
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_dmc_env
from hsuanwu.common.engine import HsuanwuEngine

train_env = make_dmc_env(env_id='cartpole_balance',
                        device='cuda:0',
                        num_envs=1,
                        resource_files=None, 
                        img_source=None,
                        total_frames=None,
                        seed=1, 
                        visualize_reward=False, 
                        from_pixels=True, 
                        frame_skip=2, frame_stack=3)

test_env = make_dmc_env(env_id='cartpole_balance',
                        device='cuda:0',
                        num_envs=1,
                        resource_files=None, 
                        img_source=None,
                        total_frames=None,
                        seed=1, 
                        visualize_reward=False, 
                        from_pixels=True, 
                        frame_skip=2, frame_stack=3)

print(train_env.observation_space)

@hydra.main(version_base=None, config_path='../cfgs/task', config_name='sac_dmc_pixel_config')
def main(cfgs):
    engine = HsuanwuEngine(cfgs=cfgs, train_env=train_env, test_env=test_env)
    engine.invoke()

if __name__ == '__main__':
    main()