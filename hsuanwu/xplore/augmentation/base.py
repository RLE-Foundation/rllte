import torch.nn as nn

from hsuanwu.common.typing import *


class BaseAugmentation(nn.Module):
    """Base class of augmentation."""

    def __init__(self) -> None:
        super().__init__()
