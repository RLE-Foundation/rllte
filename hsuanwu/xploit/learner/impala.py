import torch
from torch.nn import functional as F

from hsuanwu.common.typing import Device, Dict, Iterable, Space, Tensor
from hsuanwu.xploit import utils
from hsuanwu.xploit.learner.base import BaseLearner


class IMPALALearner(BaseLearner):
    """Importance Weighted Actor-Learner Architecture (IMPALA).
    
    Args:

    Returns:

    """
    def __init__(self, 
                 observation_space: Space, 
                 action_space: Space, 
                 action_type: str, 
                 device: Device, 
                 feature_dim: int, 
                 lr: float, 
                 eps: float) -> None:
        super().__init__(observation_space, action_space, action_type, device, feature_dim, lr, eps)