from pathlib import Path
from collections import deque
from omegaconf import open_dict

import numpy as np
import random
import hydra
import torch

from hsuanwu.common.engine import BasePolicyTrainer
from hsuanwu.common.engine import utils
from hsuanwu.common.logger import *
from hsuanwu.common.typing import *



class OnPolicyTrainer(BasePolicyTrainer):
    """Trainer for on-policy algorithms.
    
    Args:
        train_env: A Gym-like environment for training.
        test_env: A Gym-like environment for testing.
        cfgs: Dict config for configuring RL algorithms.
    
    Returns:
        On-policy trainer instance.
    """
    def __init__(self,
                 train_env: Env,
                 test_env: Env,
                 cfgs: DictConfig) -> None:
        super().__init__(train_env, test_env, cfgs)
        # xploit part
        self._learner = hydra.utils.instantiate(self._cfgs.learner)
        encoder = hydra.utils.instantiate(self._cfgs.encoder).to(self._device)
        self._learner.set_encoder(encoder)
        self._rollout_buffer = hydra.utils.instantiate(self._cfgs.buffer)

        # xplore part
        dist = hydra.utils.get_class(self._cfgs.distribution._target_)
        self._learner.set_dist(dist)
        if self._cfgs.use_aug:
            aug = hydra.utils.instantiate(self._cfgs.augmentation).to(self._device)
            self._learner.set_aug(aug)
        if self._cfgs.use_irs:
            irs = hydra.utils.instantiate(self._cfgs.reward)
            self._learner.set_irs(irs)
    
        # training track
        self._num_train_steps = self._cfgs.num_train_steps
        self._num_steps = self._cfgs.num_steps
        self._num_envs = self._cfgs.num_envs
        self._test_every_episodes = self._cfgs.test_every_episodes

        # debug
        self._logger.log(DEBUG, 'Check Accomplished. Start Training...')

    
    def train(self) -> None:
        """Training function.
        """
        episode_rewards = deque(maxlen=10)
        obs = self._train_env.reset()
        metrics = None
        # Number of updates
        num_updates = self._num_train_steps // self._num_envs // self._num_steps

        for update in range(num_updates):
            # try to test
            if update % self._test_every_episodes == 0:
                test_metrics = self.test()
                self._logger.log(level=TEST, msg=test_metrics)

            for step in range(self._num_steps):
                # sample actions
                with torch.no_grad(), utils.eval_mode(self._learner):
                    actions, values, log_probs, entropy = self._learner.act(obs, training=True, step=self._global_step)
                next_obs, rewards, dones, infos = self._train_env.step(actions)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # add transitions
                self._rollout_buffer.add(obs=obs, 
                                         actions=actions,
                                         rewards=rewards,
                                         dones=dones,
                                         log_probs=log_probs,
                                         values=values)
                
                obs = next_obs
            
            # get the value estimation of the last step
            with torch.no_grad():
                last_values = self._learner.get_value(next_obs).detach()
            
            # perform return and advantage estimation
            self._rollout_buffer.compute_returns_and_advantages(last_values)

            # policy update
            metrics = self._learner.update(self._rollout_buffer)

            # reset buffer
            self._rollout_buffer.reset()

            self._global_episode += 1
            self._global_step += self._num_envs * self._num_steps
            episode_time, total_time = self._timer.reset()

            train_metrics = {
                'step': self._global_step,
                'episode': self._global_episode,
                'episode_length': self._num_steps,
                'episode_reward': np.mean(episode_rewards),
                'fps': self._num_steps * self._num_envs / episode_time,
                'total_time': total_time
            }
            self._logger.log(level=TRAIN, msg=train_metrics)


    def test(self) -> Dict:
        """Testing function.
        """
        obs = self._test_env.reset()
        episode_rewards = list()

        while len(episode_rewards) < self._cfgs.num_test_episodes:
            with torch.no_grad(), utils.eval_mode(self._learner):
                actions, _, _, _ = self._learner.act(obs, training=False, step=self._global_step)
            next_obs, rewards, dones, infos = self._test_env.step(actions)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            
            obs = next_obs
            
        return {
            'step': self._global_step,
            'episode': self._global_episode,
            'episode_length': self._num_steps,
            'episode_reward': np.mean(episode_rewards),
            'total_time': self._timer.total_time()
        }