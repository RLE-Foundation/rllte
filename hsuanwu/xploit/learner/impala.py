import torch
import threading
from torch.nn import functional as F

from hsuanwu.common.typing import Device, Dict, Iterable, Space, Tensor
from hsuanwu.xploit import utils
from hsuanwu.xploit.learner.base import BaseLearner



def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

from torch import nn
class MinigridPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(MinigridPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0), 
            nn.init.calculate_gain('relu'))
        
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[2], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )
    
        self.fc = nn.Sequential(
            init_(nn.Linear(32, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        self.core = nn.LSTM(1024, 1024, 2)

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))


    def initial_state(self, batch_size):
        return tuple(torch.zeros(self.core.num_layers, batch_size, 
                                self.core.hidden_size) for _ in range(2))


    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs['partial_obs']
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() #/ 255.0
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        core_input = self.fc(x)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, 
                    action=action), core_state


class IMPALALearner(BaseLearner):
    """Importance Weighted Actor-Learner Architecture (IMPALA).

    Args:

    Returns:

    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_type: str,
        device: Device,
        feature_dim: int,
        lr: float = 1e-4,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(
            observation_space, action_space, action_type, device, feature_dim, lr, eps
        )

        self.actor = MinigridPolicyNet(
            observation_shape=observation_space.shape,
            num_actions=action_space.shape[0]
        ).to(self.device)
        self.actor.share_memory()

        self.learner = MinigridPolicyNet(
            observation_shape=observation_space.shape,
            num_actions=action_space.shape[0]
        ).to(self.device)

        self.opt = torch.optim.RMSprop(
            self.learner.parameters(),
            lr=self.lr,
            eps=self.eps)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, utils.lr_lambda)
    

    @staticmethod
    def update(
        actor,
        learner,
        batch,
        actor_state,
        opt,
        scheduler,
        cfgs,
        lock=threading.Lock()
    ) -> Dict[str, float]:
        """Update learner.
        """
        """Performs a learning (optimization) step."""

        with lock:
            learner_outputs, unused_state = learner(batch, actor_state)
        
            bootstrap_value = learner_outputs['baseline'][-1]

            batch = {key: tensor[1:] for key, tensor in batch.items()}
            learner_outputs = {
                key: tensor[:-1]
                for key, tensor in learner_outputs.items()
            }

            rewards = batch['reward']
            clipped_rewards = torch.clamp(rewards, -1, 1)
            
            discounts = (~batch['done']).float() * cfgs.discount

            vtrace_returns = vtrace.from_logits(
                behavior_policy_logits=batch['policy_logits'],
                target_policy_logits=learner_outputs['policy_logits'],
                actions=batch['action'],
                discounts=discounts,
                rewards=clipped_rewards,
                values=learner_outputs['baseline'],
                bootstrap_value=bootstrap_value)

            pg_loss = losses.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                                batch['action'],
                                                vtrace_returns.pg_advantages)
            baseline_loss = cfgs.learner.baseline_coef * losses.compute_baseline_loss(
                vtrace_returns.vs - learner_outputs['baseline'])
            entropy_loss = cfgs.learner.ent_coef * losses.compute_entropy_loss(
                learner_outputs['policy_logits'])

            total_loss = pg_loss + baseline_loss + entropy_loss

            episode_returns = batch['episode_return'][batch['done']]
            stats = {
                'mean_episode_return': torch.mean(episode_returns).item(),
                'total_loss': total_loss.item(),
                'pg_loss': pg_loss.item(),
                'baseline_loss': baseline_loss.item(),
                'entropy_loss': entropy_loss.item(),
            }
            
            scheduler.step()
            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(learner.parameters(), cfgs.learner.max_grad_norm)
            opt.step()

            actor.load_state_dict(learner.state_dict())
            return stats

