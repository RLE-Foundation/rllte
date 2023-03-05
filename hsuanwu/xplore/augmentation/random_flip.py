import torch

from hsuanwu.common.typing import *
from hsuanwu.xplore.augmentation.base import BaseAugmentation

class RandomFlip(BaseAugmentation):
    """Random flip operation for image augmentation.
    
    Args:

    
    Returns:

    """
    def __init__(self) -> None:
        super().__init__()