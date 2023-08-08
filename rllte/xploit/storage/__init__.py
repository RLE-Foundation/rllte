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

from .dict_replay_storage import DictReplayStorage as DictReplayStorage
from .dict_rollout_storage import DictRolloutStorage as DictRolloutStorage
from .her_replay_storage import HerReplayStorage as HerReplayStorage

# replay storage
from .nstep_replay_storage import NStepReplayStorage as NStepReplayStorage
from .prioritized_replay_storage import PrioritizedReplayStorage as PrioritizedReplayStorage

# distributed storage
from .vanilla_distributed_storage import VanillaDistributedStorage as VanillaDistributedStorage
from .vanilla_replay_storage import VanillaReplayStorage as VanillaReplayStorage

# rollout storage
from .vanilla_rollout_storage import VanillaRolloutStorage as VanillaRolloutStorage
