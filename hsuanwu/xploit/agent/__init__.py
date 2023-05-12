import os

from omegaconf import OmegaConf

from .base import BaseAgent as BaseAgent
from .daac import DAAC as DAAC
from .drqv2 import DrQv2 as DrQv2
from .impala import IMPALA as IMPALA
from .ppg import PPG as PPG
from .ppo import PPO as PPO
from .sac import SAC as SAC

dir_name = os.path.dirname(os.path.realpath(__file__))

ALL_DEFAULT_CFGS = {
    "DAAC": OmegaConf.load(os.path.join(dir_name, "daac.yaml")),
    "DrQv2": OmegaConf.load(os.path.join(dir_name, "drqv2.yaml")),
    "SAC": OmegaConf.load(os.path.join(dir_name, "sac.yaml")),
    "PPO": OmegaConf.load(os.path.join(dir_name, "ppo.yaml")),
    "PPG": OmegaConf.load(os.path.join(dir_name, "ppg.yaml")),
    "IMPALA": OmegaConf.load(os.path.join(dir_name, "impala.yaml")),
}
