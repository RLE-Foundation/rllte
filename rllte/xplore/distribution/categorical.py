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


import torch as th
import torch.distributions as pyd

from rllte.common.prototype import BaseDistribution


class Categorical(BaseDistribution):
    """Categorical distribution for sampling actions for 'Discrete' tasks.

    Args:
        logits (th.Tensor): The event log probabilities (unnormalized).

    Returns:
        Categorical distribution instance.
    """

    def __init__(self, logits: th.Tensor) -> None:
        super().__init__()
        self.dist = pyd.Categorical(logits=logits)

    @property
    def probs(self) -> th.Tensor:
        """Return probabilities."""
        return self.dist.probs

    @property
    def logits(self) -> th.Tensor:
        """Returns the unnormalized log probabilities."""
        return self.dist.logits

    def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # noqa B008
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
        return self.dist.sample()

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Returns the log of the probability density/mass function evaluated at actions.

        Args:
            actions (th.Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
        return self.dist.log_prob(actions)

    def entropy(self) -> th.Tensor:
        """Returns the Shannon entropy of distribution."""
        return self.dist.entropy()

    @property
    def mode(self) -> th.Tensor:
        """Returns the mode of the distribution."""
        return self.dist.probs.argmax(axis=-1)

    @property
    def mean(self) -> th.Tensor:
        """Returns the mean of the distribution."""
        return self.dist.probs.argmax(axis=-1)
