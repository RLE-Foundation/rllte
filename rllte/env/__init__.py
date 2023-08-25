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


from .utils import make_rllte_env as make_rllte_env

try:
    from .atari import make_atari_env as make_atari_env
    from .atari import make_envpool_atari_env as make_envpool_atari_env
except Exception:
    pass

try:
    from .bullet import make_bullet_env as make_bullet_env
except Exception:
    pass

try:
    from .dmc import make_dmc_env as make_dmc_env
except Exception:
    pass

try:
    from .minigrid import make_minigrid_env as make_minigrid_env
except Exception:
    pass

try:
    from .multibinary import make_multibinary_env as make_multibinary_env
except Exception:
    pass

try:
    from .multidiscrete import make_multidiscrete_env as make_multidiscrete_env
except Exception:
    pass

try:
    from .procgen import make_procgen_env as make_procgen_env
    from .procgen import make_envpool_procgen_env as make_envpool_procgen_env
except Exception:
    pass

try:
    from .bitflipping import make_bitflipping_env as make_bitflipping_env
except Exception:
    pass