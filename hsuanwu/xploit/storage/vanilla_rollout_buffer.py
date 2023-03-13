from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import torch

from hsuanwu.common.typing import *

class VanillaRolloutBuffer:
    """Vanilla rollout buffer for on-policy algorithms.
    
    Args:
        obs_shape: The data shape of observations.
        action_shape: The data shape of actions.
        action_type: The type of actions, 'cont' or 'dis'.
        num_steps: The sample steps of per rollout.
        num_envs: The number of parallel environments.

    Returns:
        Vanilla rollout buffer.
    """
    def __init__(self,
                 device: torch.device,
                 obs_shape: Tuple,
                 action_shape: Tuple,
                 action_type: str,
                 num_steps: int,
                 num_envs: int) -> None:
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._device = torch.device(device)

        self._storage = dict()
        # transition part
        self._storage['obs'] = torch.empty(size=(num_steps, num_envs, *obs_shape), 
                                           dtype=torch.float32, 
                                           device=self._device)
        if action_type == 'dis':
            self._storage['actions']  = torch.empty(size=(num_steps, num_envs, 1), 
                                                    dtype=torch.float32, 
                                                    device=self._device)
        if action_type == 'cont':
            self._storage['actions'] = torch.empty(size=(num_steps, num_envs, action_shape[0]), 
                                                   dtype=torch.float32, 
                                                   device=self._device)
        self._storage['rewards'] = torch.empty(size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device)
        self._storage['dones'] = torch.empty(size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device)
        # extra part
        self._storage['log_probs'] = torch.empty(size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device)
        self._storage['values'] = torch.empty(size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device)
        self._storage['returns'] = torch.empty(size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device)

        self._global_step = 0


    def add(self, obs, actions, rewards, dones, log_probs, values) -> None:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self._device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self._device)
        log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=self._device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self._device)

        self._storage['obs'][self._global_step].copy_(obs)
        self._storage['actions'][self._global_step].copy_(actions)
        self._storage['rewards'][self._global_step].copy_(rewards)
        self._storage['dones'][self._global_step].copy_(dones)
        self._storage['log_probs'][self._global_step].copy_(log_probs)
        self._storage['values'][self._global_step].copy_(values)

        self._global_step = (self._global_step + 1) % self._num_steps

    def sample(self,):
        pass

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda):
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


