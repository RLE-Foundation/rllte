import collections
import threading

import torch
from torch.nn import functional as F

from hsuanwu.common.typing import Device, Dict, DictConfig, Iterable, Space, Tensor
from hsuanwu.xploit.learner.base import BaseLearner
from hsuanwu.xploit.learner.utils import lr_lambda


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


from torch import nn


class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


class VTrace(object):
    """Compute V-trace off-policy actor critic targets."""

    def __init__(self) -> None:
        pass

    def action_log_probs(self, policy_logits, actions):
        return -F.nll_loss(
            F.log_softmax(torch.flatten(policy_logits, 0, 1), dim=-1),
            torch.flatten(actions, 0, 1),
            reduction="none",
        ).view_as(actions)

    def from_logits(
        self,
        behavior_policy_logits,
        target_policy_logits,
        actions,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
    ):
        """V-trace for softmax policies."""

        target_action_log_probs = self.action_log_probs(target_policy_logits, actions)
        behavior_action_log_probs = self.action_log_probs(
            behavior_policy_logits, actions
        )
        log_rhos = target_action_log_probs - behavior_action_log_probs
        vtrace_returns = self.from_importance_weights(
            log_rhos=log_rhos,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold,
        )
        return VTraceFromLogitsReturns(
            log_rhos=log_rhos,
            behavior_action_log_probs=behavior_action_log_probs,
            target_action_log_probs=target_action_log_probs,
            **vtrace_returns._asdict(),
        )

    @torch.no_grad()
    def from_importance_weights(
        self,
        log_rhos,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
    ):
        """V-trace from log importance weights."""
        with torch.no_grad():
            rhos = torch.exp(log_rhos)
            if clip_rho_threshold is not None:
                clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
            else:
                clipped_rhos = rhos

            cs = torch.clamp(rhos, max=1.0)
            # Append bootstrapped value to get [v1, ..., v_t+1]
            values_t_plus_1 = torch.cat(
                [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
            )
            deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

            acc = torch.zeros_like(bootstrap_value)
            result = []
            for t in range(discounts.shape[0] - 1, -1, -1):
                acc = deltas[t] + discounts[t] * cs[t] * acc
                result.append(acc)
            result.reverse()
            vs_minus_v_xs = torch.stack(result)

            # Add V(x_s) to get v_s.
            vs = torch.add(vs_minus_v_xs, values)

            # Advantage for policy gradient.
            vs_t_plus_1 = torch.cat(
                [vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
            )
            if clip_pg_rho_threshold is not None:
                clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
            else:
                clipped_pg_rhos = rhos
            pg_advantages = clipped_pg_rhos * (
                rewards + discounts * vs_t_plus_1 - values
            )

            # Make sure no gradients backpropagated through the returned values.
            return VTraceReturns(vs=vs, pg_advantages=pg_advantages)


class IMPALALearner(BaseLearner):
    """Importance Weighted Actor-Learner Architecture (IMPALA).

    Args:
        observation_space (Space): Observation space of the environment.
        action_space (Space): Action shape of the environment.
        action_type (str): Continuous or discrete action. "cont" or "dis".
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

        use_lstm (bool): .
        ent_coef (float): Weighting coefficient of entropy bonus.
        baseline_coef(float): .
        max_grad_norm (float): Maximum norm of gradients.
        discount (float): Discount factor.
    Returns:
        IMPALALearner distance.
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_type: str,
        device: Device,
        feature_dim: int,
        lr: float = 0.0004,
        eps: float = 0.01,
        use_lstm: bool = False,
        ent_coef: float = 0.01,
        baseline_coef: float = 0.5,
        max_grad_norm: float = 40,
        discount: float = 0.99,
    ) -> None:
        super().__init__(
            observation_space, action_space, action_type, device, feature_dim, lr, eps
        )

        self.ent_coef = ent_coef
        self.baseline_coef = baseline_coef
        self.max_grad_norm = max_grad_norm
        self.discount = discount

        self.actor = AtariNet(
            observation_shape=observation_space.shape,
            num_actions=action_space.shape[0],
            use_lstm=use_lstm,
        )
        self.actor.share_memory()

        self.learner = AtariNet(
            observation_shape=observation_space.shape,
            num_actions=action_space.shape[0],
            use_lstm=use_lstm,
        ).to(self.device)

        self.opt = torch.optim.RMSprop(
            self.learner.parameters(), lr=self.lr, eps=self.eps
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    def train(self, training: bool = True) -> None:
        """Set the train mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self.actor.train(training)
        self.learner.train(training)

    @staticmethod
    def update(
        cfgs: DictConfig,
        actor_model,
        learner_model,
        batch,
        init_actor_state,
        optimizer,
        lr_scheduler,
        lock=threading.Lock(),
    ):
        ###########################################################################
        def compute_policy_gradient_loss(logits, actions, advantages):
            cross_entropy = F.nll_loss(
                F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
                target=torch.flatten(actions, 0, 1),
                reduction="none",
            )
            cross_entropy = cross_entropy.view_as(advantages)
            advantages.requires_grad = False
            policy_gradient_loss_per_timestep = cross_entropy * advantages
            return torch.sum(torch.mean(policy_gradient_loss_per_timestep, dim=1))

        def compute_baseline_loss(advantages):
            return 0.5 * torch.sum(advantages**2)

        def compute_entropy_loss(logits):
            """Return the entropy loss, i.e., the negative entropy of the policy."""
            policy = F.softmax(logits, dim=-1)
            log_policy = F.log_softmax(logits, dim=-1)
            return torch.sum(policy * log_policy)

        ###########################################################################
        """Performs a learning (optimization) step."""
        with lock:
            learner_outputs, _ = learner_model(batch, init_actor_state)

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = {key: tensor[1:] for key, tensor in batch.items()}
            learner_outputs = {
                key: tensor[:-1] for key, tensor in learner_outputs.items()
            }

            rewards = batch["reward"]
            clipped_rewards = torch.clamp(rewards, -1, 1)

            discounts = (~batch["done"]).float() * cfgs.discount

            vtrace_returns = VTrace().from_logits(
                behavior_policy_logits=batch["policy_logits"],
                target_policy_logits=learner_outputs["policy_logits"],
                actions=batch["action"],
                discounts=discounts,
                rewards=clipped_rewards,
                values=learner_outputs["baseline"],
                bootstrap_value=bootstrap_value,
            )

            pg_loss = compute_policy_gradient_loss(
                learner_outputs["policy_logits"],
                batch["action"],
                vtrace_returns.pg_advantages,
            )
            baseline_loss = cfgs.learner.baseline_coef * compute_baseline_loss(
                vtrace_returns.vs - learner_outputs["baseline"]
            )
            entropy_loss = cfgs.learner.ent_coef * compute_entropy_loss(
                learner_outputs["policy_logits"]
            )

            total_loss = pg_loss + baseline_loss + entropy_loss

            episode_returns = batch["episode_return"][batch["done"]]

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                learner_model.parameters(), cfgs.learner.max_grad_norm
            )
            optimizer.step()
            lr_scheduler.step()

            actor_model.load_state_dict(learner_model.state_dict())
            return {
                "episode_returns": tuple(episode_returns.cpu().numpy()),
                "mean_episode_return": torch.mean(episode_returns).item(),
                "total_loss": total_loss.item(),
                "pg_loss": pg_loss.item(),
                "baseline_loss": baseline_loss.item(),
                "entropy_loss": entropy_loss.item(),
            }
