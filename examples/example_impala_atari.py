import os
import sys
import hydra

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

# from hsuanwu.env import make_dmc_env
# from hsuanwu.common.engine import DistributedTrainer
from hsuanwu.common.engine.distributed_trainer import train

# @hydra.main(version_base=None, config_path='../cfgs/task', config_name='impala_atari_config')
# def main(cfgs):
#     trainer = DistributedTrainer(train_env=None, test_env=None, cfgs=cfgs)
#     trainer.train()

import omegaconf
def main(cfgs):
    train(cfgs)

if __name__ == '__main__':
    cfgs = omegaconf.OmegaConf.load('./cfgs/task/impala_atari_config.yaml')
    main(cfgs)
    # trainer = DistributedTrainer(train_env=None, test_env=None, cfgs=cfgs)
    # trainer.train()