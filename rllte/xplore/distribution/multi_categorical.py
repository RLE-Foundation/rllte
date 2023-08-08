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

from typing import Tuple

import torch as th
import torch.distributions as pyd

from rllte.common.prototype import BaseDistribution


class MultiCategorical(BaseDistribution):
    """Categorical distribution for sampling actions for 'MultiDiscrete' tasks.

    Args:
        logits (Tuple[th.Tensor, ...]): The event log probabilities (unnormalized).

    Returns:
        Categorical distribution instance.
    """

    def __init__(self, logits: Tuple[th.Tensor, ...]) -> None:
        super().__init__()
        self.dist = [pyd.Categorical(logits=logits_) for logits_ in logits]

    @property
    def probs(self) -> Tuple[th.Tensor, ...]: 
        """Return probabilities."""
        return (dist.probs for dist in self.dist) # type: ignore

    @property
    def logits(self) -> Tuple[th.Tensor, ...]: 
        """Returns the unnormalized log probabilities."""
        return (dist.logits for dist in self.dist) # type: ignore

    def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # noqa B008
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        return th.stack([dist.sample() for dist in self.dist], dim=1)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Returns the log of the probability density/mass function evaluated at actions.

        Args:
            actions (th.Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
        return th.stack([dist.log_prob(action) for dist, action in zip(self.dist, th.unbind(actions, dim=1))], dim=1).sum(
            dim=1
        )

    def entropy(self) -> th.Tensor:
        """Returns the Shannon entropy of distribution."""
        return th.stack([dist.entropy() for dist in self.dist], dim=1).sum(dim=1)

    @property
    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        return th.stack([dist.probs.argmax(axis=-1) for dist in self.dist], dim=1)

    @property
    def mean(self) -> th.Tensor:
        """Returns the mean of the distribution."""
        return th.stack([dist.probs.argmax(axis=-1) for dist in self.dist], dim=1)
