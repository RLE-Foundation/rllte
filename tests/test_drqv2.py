import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from pathlib import Path
from hsuanwu.common.logger import Logger

# env part
from hsuanwu.envs import make_dmc_env, FrameStack

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
                       frame_skip=2)
env = FrameStack(env, k=3)

class OffPolicyTrainer:
    def __init__(self, train_env, eval_env, cfgs) -> None:
        # setup
        self.cfgs = cfgs
        self.train_env = train_env
        self.eval_env = eval_env
        cfgs.observation_space = {'shape': train_env.observation_space.shape}
        if cfgs.action_type == 'cont':
            cfgs.action_space = {'shape': train_env.action_space.shape}
        elif cfgs.action_type == 'dis':
            cfgs.action_space = {'shape': train_env.action_space.n}
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
        
        self.global_step = 0
        self.global_episode = 0
        self.train_until_step = cfgs.num_train_frames // cfgs.action_repeat
        self.train_start_step = cfgs.num_seed_frames // cfgs.action_repeat
    
    def train(self):
        episode_step, episode_reward = 0, 0
        obs = self.train_env.reset()

        while self.global_step <= self.train_until_step:
            # try to evaluate
            eval_metrics = self.eval()
            print(self.global_step, eval_metrics)

            # sample actions
            with torch.no_grad():
                action = self.learner.act(obs, training=True, step=self.global_step)
            next_obs, reward, done, info = self.train_env.step(action)
            episode_reward += reward
            episode_step += 1

            # save transitions
            self.buffer.add(obs, action, reward, done)

            # training
            if self.global_step >= self.train_start_step:
                self.learner.update(self.loader, step=self.global_step)

            if done:
                obs = self.train_env.reset()
                self.global_episode += 1
                episode_step, episode_reward = 0, 0

            obs = next_obs
    
    def eval(self):
        step, episode, total_reward = 0, 0, 0
        while episode <= self.cfgs.num_eval_episodes:
            obs = self.eval_env.reset()
            with torch.no_grad():
                action = self.learner.act(obs, training=False, step=self.global_step)

            next_obs, reward, done, info = self.train_env.step(action)
            total_reward += reward
            step += 1

            if done:
                obs = self.eval_env.reset()
                episode += 1
            
            obs = next_obs
        
        return {'episode_reward': total_reward / episode, 
                'episode_length': step * self.cfgs.action_repeat / episode}

@hydra.main(version_base=None, config_path='../cfgs', config_name='config')
def main(cfgs):
    trainer = OffPolicyTrainer(env, cfgs)
    trainer.train()

if __name__ == '__main__':
    main()