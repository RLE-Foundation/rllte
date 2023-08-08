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

# primitives
from .base_agent import BaseAgent as BaseAgent
from .base_augmentation import BaseAugmentation as BaseAugmentation
from .base_distribution import BaseDistribution as BaseDistribution
from .base_encoder import BaseEncoder as BaseEncoder
from .base_policy import BasePolicy as BasePolicy
from .base_reward import BaseIntrinsicRewardModule as BaseIntrinsicRewardModule
from .base_storage import BaseStorage as BaseStorage
# agent prototypes
from .distributed_agent import DistributedAgent as DistributedAgent
from .off_policy_agent import OffPolicyAgent as OffPolicyAgent
from .on_policy_agent import OnPolicyAgent as OnPolicyAgent