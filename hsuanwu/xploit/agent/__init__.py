from .base import BaseAgent as BaseAgent
from .daac import DAAC as DAAC
from .daac import DEFAULT_CFGS as DAAC_DEFAULT_CFGS
from .daac import MATCH_KEYS as DAAC_MATCH_KEYS
from .drqv2 import DEFAULT_CFGS as DRQV2_DEFAULT_CFGS
from .drqv2 import MATCH_KEYS as DRQV2_MATCH_KEYS
from .drqv2 import DrQv2 as DrQv2
from .impala import DEFAULT_CFGS as IMPALA_DEFAULT_CFGS
from .impala import IMPALA as IMPALA
from .impala import MATCH_KEYS as IMPALA_MATCH_KEYS
from .ppg import DEFAULT_CFGS as PPG_DEFAULT_CFGS
from .ppg import MATCH_KEYS as PPG_MATCH_KEYS
from .ppg import PPG as PPG
from .ppo import DEFAULT_CFGS as PPO_DEFAULT_CFGS
from .ppo import MATCH_KEYS as PPO_MATCH_KEYS
from .ppo import PPO as PPO
from .sac import DEFAULT_CFGS as SAC_DEFAULT_CFGS
from .sac import MATCH_KEYS as SAC_MATCH_KEYS
from .sac import SAC as SAC

ALL_DEFAULT_CFGS = {
    "DAAC": DAAC_DEFAULT_CFGS,
    "DrQv2": DRQV2_DEFAULT_CFGS,
    "PPG": PPG_DEFAULT_CFGS,
    "SAC": SAC_DEFAULT_CFGS,
    "PPO": PPO_DEFAULT_CFGS,
    "IMPALA": IMPALA_DEFAULT_CFGS,
}
ALL_MATCH_KEYS = {
    "DAAC": DAAC_MATCH_KEYS,
    "DrQv2": DRQV2_MATCH_KEYS,
    "PPG": PPG_MATCH_KEYS,
    "SAC": SAC_MATCH_KEYS,
    "PPO": PPO_MATCH_KEYS,
    "IMPALA": IMPALA_MATCH_KEYS,
}
