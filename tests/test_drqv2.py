import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from pathlib import Path
from hsuanwu.common.logger import Logger

# env part
from hsuanwu.envs import make_dmc_env

import hydra
import torch

env = make_dmc_env(domain_name='hopper', 
                       task_name='hop', 
                       resource_files=None, 
                       img_source=None,
                       total_frames=None,
                       seed=1, 
                       visualize_reward=False, 
                       from_pixels=True, 
                       frame_skip=1)

class OffPolicyTrainer:
    def __init__(self, env, cfgs) -> None:
        cfgs.observation_space = {'shape': env.observation_space.shape}
        if cfgs.action_type == 'cont':
            cfgs.action_space = {'shape': env.action_space.shape}
        elif cfgs.action_type == 'dis':
            cfgs.action_space = {'shape': env.action_space.n}
        self.device = torch.device(cfgs.device)

        # xploit part
        self.learner = hydra.utils.instantiate(cfgs.learner)
        self.learner.encoder = hydra.utils.instantiate(cfgs.encoder).to(self.device)
        self.learner.encoder_opt = torch.optim.Adam(
            self.learner.encoder.parameters(), lr=cfgs.learner.lr)
        self.buffer = hydra.utils.instantiate(cfgs.buffer)

        # xplore part
        self.learner.dist = hydra.utils.get_class(cfgs.distribution._target_)
        if cfgs.use_aug:
            self.learner.aug = hydra.utils.instantiate(cfgs.augmentation)
        if cfgs.use_irs:
            self.learner.reward = hydra.utils.instantiate(cfgs.reward)
        
        self.loader = torch.utils.data.DataLoader(self.buffer,
                                                  batch_size=cfgs.buffer.batch_size,
                                                  num_workers=cfgs.buffer.num_workers,
                                                  pin_memory=cfgs.pin_memory)
    
    def train(self):
        


@hydra.main(version_base=None, config_path='../cfgs', config_name='config')
def main(cfgs):
    trainer = OffPolicyTrainer(env, cfgs)
    trainer.train()

if __name__ == '__main__':
    main()