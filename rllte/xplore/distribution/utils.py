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


import re

import numpy as np


def schedule(schdl: str, step: int) -> float:
    """Exploration noise schedule.

    Args:
        schdl (str): Schedule mode.
        step (int): global training step.

    Returns:
        Standard deviation.
    """

    match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
    if match:
        init, final, duration = (float(g) for g in match.groups())
        mix = np.clip(step / duration, 0.0, 1.0)
        return (1.0 - mix) * init + mix * final
    match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
    if match:
        init, final1, duration1, final2, duration2 = (float(g) for g in match.groups())
        if step <= duration1:
            mix = np.clip(step / duration1, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final1
        else:
            mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
            return (1.0 - mix) * final1 + mix * final2
