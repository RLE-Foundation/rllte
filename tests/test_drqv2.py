import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from pathlib import Path
from hsuanwu.common.logger import Logger

# env part
from hsuanwu.envs import make_dmc_env
# xploit part
from hsuanwu.xploit.encoder import VanillaCnnEncoder
from hsuanwu.xploit.learner import ContinuousLearner
from hsuanwu.xploit.storage import NStepReplayBuffer
# xplore part
from hsuanwu.xplore.distribution import TruncatedNormal
from hsuanwu.xplore.augmentation import RandomShift

import hydra

env = make_dmc_env(domain_name='hopper', 
                       task_name='hop', 
                       resource_files=None, 
                       img_source=None,
                       total_frames=None,
                       seed=1, 
                       visualize_reward=False, 
                       from_pixels=True, 
                       frame_skip=1)

class OffPolicyTrainer(object):
    def __init__(self, cfgs) -> None:
        cfgs.encoder.observation_space = env.observation_space
        self.encoder = hydra.utils.instantiate(cfgs.encoder)
        print(self.encoder)
        

@hydra.main(version_base=None, config_path='../cfgs', config_name='config')
def main(cfgs):
    trainer = OffPolicyTrainer(cfgs)

if __name__ == '__main__':
    main()