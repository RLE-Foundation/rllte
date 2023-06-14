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


from typing import Dict, List, Optional, Tuple, Union

import arch.bootstrap as arch_bs
import numpy as np
from numpy import random

Float = Union[float, np.float32, np.float64]


def min_max_normalize(value: np.ndarray, min_scores: np.ndarray, max_scores: np.ndarray) -> np.ndarray:
    """Perform `Max-Min` normalization."""
    return (value - min_scores) / (max_scores - min_scores)


class StratifiedBootstrap(arch_bs.IIDBootstrap):
    """Bootstrap using stratified resampling.
        Borrowed from: https://github.com/google-research/rliable/blob/master/rliable/library.py

    Supports numpy arrays. Data returned has the same type as the input data.
    Data entered using keyword arguments is directly accessibly as an attribute.

    To ensure a reproducible bootstrap, you must set the `random_state`
    attribute after the bootstrap has been created. See the example below.
    Note that `random_state` is a reserved keyword and any variable
    passed using this keyword must be an instance of `RandomState`.

    Examples
    --------
    Data can be accessed in a number of ways.  Positional data is retained in
    the same order as it was entered when the bootstrap was initialized.
    Keyword data is available both as an attribute or using a dictionary syntax
    on kw_data.

    >>> from rliable.library import StratifiedBootstrap
    >>> from numpy.random import standard_normal
    >>> x = standard_normal((5, 50))
    >>> bs = StratifiedBootstrap(x)
    >>> for data in bs.bootstrap(100):
    ...     bs_x = data[0][0]
    >>> bs.conf_int(np.mean, method='percentile', reps=50000)  # 95% CIs for mean

    Set the random_state if reproducibility is required.

    >>> from numpy.random import RandomState
    >>> rs = RandomState(1234)
    >>> bs = StratifiedBootstrap(x, random_state=rs)

    See also: `arch.bootstrap.IIDBootstrap`

    Attributes:
        data: tuple, Two-element tuple with the pos_data in the first position and
            kw_data in the second (pos_data, kw_data). Derived from `IIDBootstrap`.
        pos_data: tuple, Tuple containing the positional arguments (in the order
            entered). Derived from `IIDBootstrap`.
        kw_data: dict, Dictionary containing the keyword arguments. Derived from
            `IIDBootstrap`.
    """

    _name = "Stratified Bootstrap"

    def __init__(
        self,
        *args: np.ndarray,
        random_state: Optional[random.RandomState] = None,
        task_bootstrap: bool = False,
        **kwargs: np.ndarray,
    ) -> None:
        """Initializes StratifiedBootstrap.

        Args:
            *args: Positional arguments to bootstrap. Typically used for the
                performance on a suite of tasks with multiple runs/episodes. The inputs
                are assumed to be of the shape `(num_runs, num_tasks, ..)`.
            random_state: If specified, ensures reproducibility in uncertainty
                estimates.
            task_bootstrap: Whether to perform bootstrapping (a) over runs or (b) over
                both runs and tasks. Defaults to False which corresponds to (a). (a)
                captures the statistical uncertainty in the aggregate performance if the
                experiment is repeated using a different set of runs (e.g., changing
                seeds) on the same set of tasks. (b) captures the sensitivity of the
                aggregate performance to a given task and provides the performance
                estimate if we had used a larger unknown population of tasks.
            **kwargs: Keyword arguments, passed directly to `IIDBootstrap`.
        """

        super().__init__(*args, random_state=random_state, **kwargs)
        self._args_shape = args[0].shape
        self._num_tasks = self._args_shape[1]
        self._parameters = [self._num_tasks, task_bootstrap]
        self._task_bootstrap = task_bootstrap
        self._strata_indices = self._get_strata_indices()

    def _get_strata_indices(self) -> List[np.ndarray]:
        """Samples partial indices for bootstrap resamples.

        Returns:
            A list of arrays of size N x 1 x 1 x .., 1 x M x 1 x ..,
            1 x 1 x L x .. and so on, where the `args_shape` is `N x M x L x ..`.
        """
        ogrid_indices = tuple(slice(x) for x in (0, *self._args_shape[1:]))
        strata_indices = np.ogrid[ogrid_indices]
        return strata_indices[1:]

    def update_indices(
        self,
    ) -> Tuple[np.ndarray, ...]:
        """Selects the indices to sample from the bootstrap distribution."""
        # `self._num_items` corresponds to the number of runs
        indices = np.random.choice(self._num_items, self._args_shape, replace=True)
        if self._task_bootstrap:
            task_indices = np.random.choice(self._num_tasks, self._strata_indices[0].shape, replace=True)
            return (indices, task_indices, *self._strata_indices[1:])
        return (indices, *self._strata_indices)


class StratifiedIndependentBootstrap(arch_bs.IndependentSamplesBootstrap):
    """Stratified Bootstrap where each input is independently resampled.
        Borrowed from: https://github.com/google-research/rliable/blob/master/rliable/library.py

    This bootstrap is useful for computing CIs for metrics which take multiple
    score arrays, possibly with different number of runs, as input, such as
    average probability of improvement. See also: `StratifiedBootstrap` and
    `arch_bs.IndependentSamplesBootstrap`.

    Attributes:
        data: tuple, Two-element tuple with the pos_data in the first position and
            kw_data in the second (pos_data, kw_data). Derived from
            `IndependentSamplesBootstrap`.
        pos_data: tuple, Tuple containing the positional arguments (in the order
            entered). Derived from `IndependentSamplesBootstrap`.
        kw_data: dict, Dictionary containing the keyword arguments. Derived from
            `IndependentSamplesBootstrap`.
    """

    def __init__(
        self,
        *args: np.ndarray,
        random_state: Optional[random.RandomState] = None,
        **kwargs: np.ndarray,
    ) -> None:
        """Initializes StratifiedIndependentSamplesBootstrap.

        Args:
            *args: Positional arguments to bootstrap. Typically used for the
                performance on a suite of tasks with multiple runs/episodes. The inputs
                are assumed to be of the shape `(num_runs, num_tasks, ..)`.
            random_state: If specified, ensures reproducibility in uncertainty
                estimates.
            **kwargs: Keyword arguments, passed directly to `IIDBootstrap`.
        """

        super().__init__(*args, random_state=random_state, **kwargs)
        self._args_shapes = [arg.shape for arg in args]
        self._kwargs_shapes = {key: val.shape for key, val in self._kwargs.items()}
        self._args_strata_indices = [self._get_strata_indices(arg_shape) for arg_shape in self._args_shapes]
        self._kwargs_strata_indices = {
            key: self._get_strata_indices(kwarg_shape) for key, kwarg_shape in self._kwargs_shapes.items()
        }

    def _get_strata_indices(self, array_shape: Tuple[int, ...]) -> List[np.ndarray]:
        """Samples partial indices for bootstrap resamples.

        Args:
            array_shape: Shape of array for which strata indices are created.

        Returns:
            A list of arrays of size N x 1 x 1 x .., 1 x M x 1 x ..,
            1 x 1 x L x .. and so on, where the `array_shape` is `N x M x L x ..`.
        """
        ogrid_indices = tuple(slice(x) for x in (0, *array_shape[1:]))
        strata_indices = np.ogrid[ogrid_indices]
        return strata_indices[1:]

    def _get_indices(
        self, num_runs: int, array_shape: Tuple[int, ...], strata_indices: List[np.ndarray]
    ) -> Tuple[np.ndarray, ...]:
        """Helper function for updating bootstrap indices."""
        indices = np.random.choice(num_runs, array_shape, replace=True)
        return (indices, *strata_indices)

    def update_indices(
        self,
    ) -> Tuple[List[Tuple[np.ndarray, ...]], Dict[str, Tuple[np.ndarray, ...]]]:
        """Update independent sampling indices for the next bootstrap iteration."""

        pos_indices = [
            self._get_indices(self._num_arg_items[i], self._args_shapes[i], self._args_strata_indices[i])
            for i in range(self._num_args)
        ]
        kw_indices = {}
        for key in self._kwargs:
            kw_indices[key] = self._get_indices(
                self._num_kw_items[key], self._kwargs_shapes[key], self._kwargs_strata_indices[key]
            )
        return pos_indices, kw_indices
