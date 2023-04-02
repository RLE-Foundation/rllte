import torch

from hsuanwu.common.typing import ABC, Device, Dict, Space, abstractmethod


class BaseLearner(ABC):
    """Base class of learner.

    Args:
        observation_space (Space): Observation space of the environment.
        action_space (Space): Action shape of the environment.
        action_type (str): Continuous or discrete action. "cont" or "dis".
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

    Returns:
        Base learner instance.
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_type: str,
        device: Device,
        feature_dim: int,
        lr: float,
        eps: float,
    ) -> None:
        self.obs_space = observation_space
        self.action_space = action_space
        self.action_type = action_type
        self.device = torch.device(device)
        self.feature_dim = feature_dim
        self.lr = lr
        self.eps = eps

        # placeholder for distribution, augmentation, and intrinsic reward function.
        self.encoder = None
        self.encoder_opt = None
        self.dist = None
        self.aug = None
        self.irs = None

    @abstractmethod
    def train(self, training: bool = True) -> None:
        """Set the train mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training

    @abstractmethod
    def update(self, *args) -> Dict[str, float]:
        """Update learner.

        Args:


        Returns:
            Training metrics such as loss functions.
        """
