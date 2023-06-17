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


from .gaussian_noise import GaussianNoise as GaussianNoise
from .grayscale import GrayScale as GrayScale
from .identity import Identity as Identity
from .random_amplitude_scaling import RandomAmplitudeScaling as RandomAmplitudeScaling
from .random_colorjitter import RandomColorJitter as RandomColorJitter
from .random_convolution import RandomConvolution as RandomConvolution
from .random_crop import RandomCrop as RandomCrop
from .random_cutout import RandomCutout as RandomCutout
from .random_cutoutcolor import RandomCutoutColor as RandomCutoutColor
from .random_flip import RandomFlip as RandomFlip
from .random_rotate import RandomRotate as RandomRotate
from .random_shift import RandomShift as RandomShift
from .random_translate import RandomTranslate as RandomTranslate
