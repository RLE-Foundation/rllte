

```python
import os
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_atari_env
from hsuanwu.common.engine import OnPolicyTrainer

train_env = make_atari_env(
    env_id='Alien-v5',
    num_envs=8,
    seed=1,
    frame_stack=4,
    device='cuda'
)

test_env = make_atari_env(
    env_id='Alien-v5',
    num_envs=8,
    seed=1,
    frame_stack=4,
    device='cuda'
)

@hydra.main(version_base=None, config_path='../cfgs', config_name='discrete_task_config')
def main(cfgs):
    trainer = OnPolicyTrainer(train_env=train_env, test_env=test_env, cfgs=cfgs)
    trainer.train()

if __name__ == '__main__':
    main()```