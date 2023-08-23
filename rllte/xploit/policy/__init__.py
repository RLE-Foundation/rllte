# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

# distributed
from .distributed_actor_learner import DistributedActorLearner as DistributedActorLearner
# off-policy
from .off_policy_double_qnetwork import OffPolicyDoubleQNetwork as OffPolicyDoubleQNetwork
from .off_policy_stoch_actor_double_critic import OffPolicyStochActorDoubleCritic as OffPolicyStochActorDoubleCritic
from .off_policy_det_actor_double_critic import OffPolicyDetActorDoubleCritic as OffPolicyDetActorDoubleCritic
# on-policy
from .on_policy_shared_actor_critic import OnPolicySharedActorCritic as OnPolicySharedActorCritic
from .on_policy_decoupled_actor_critic import OnPolicyDecoupledActorCritic as OnPolicyDecoupledActorCritic
