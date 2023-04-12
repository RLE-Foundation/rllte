import torch

from hsuanwu.common.typing import ABC, Device, Dict, abstractmethod


class BaseLearner(ABC):
    """Base class of learner.

    Args:
        observation_space (Dict): Observation space of the environment.
            For supporting Hydra, the original 'observation_space' is transformed into a dict like {"shape": observation_space.shape, }.
        action_space (Dict): Action shape of the environment.
            For supporting Hydra, the original 'action_space' is transformed into a dict like
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted by the encoder.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

    Returns:
        Base learner instance.
    """

    def __init__(
        self,
        observation_space: Dict,
        action_space: Dict,
        device: Device,
        feature_dim: int,
        lr: float,
        eps: float,
    ) -> None:
        self.obs_space = observation_space
        self.action_space = action_space
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
