from .base import BaseLearner as BaseLearner
from .drqv2 import DEFAULT_CFGS as DRQV2_DEFAULT_CFGS
from .drqv2 import MATCH_KEYS as DRQV2_MATCH_KEYS
from .drqv2 import DrQv2Learner as DrQv2Learner
from .impala import DEFAULT_CFGS as IMPALA_DEFAULT_CFGS
from .impala import MATCH_KEYS as IMPALA_MATCH_KEYS
from .impala import IMPALALearner as IMPALALearner
from .network import *
from .ppg import DEFAULT_CFGS as PPG_DEFAULT_CFGS
from .ppg import MATCH_KEYS as PPG_MATCH_KEYS
from .ppg import PPGLearner as PPGLearner
from .ppo import DEFAULT_CFGS as PPO_DEFAULT_CFGS
from .ppo import MATCH_KEYS as PPO_MATCH_KEYS
from .ppo import PPOLearner as PPOLearner
from .sac import DEFAULT_CFGS as SAC_DEFAULT_CFGS
from .sac import MATCH_KEYS as SAC_MATCH_KEYS
from .sac import SACLearner as SACLearner

ALL_DEFAULT_CFGS = {
    "DrQv2Learner": DRQV2_DEFAULT_CFGS,
    "PPGLearner": PPG_DEFAULT_CFGS,
    "SACLearner": SAC_DEFAULT_CFGS,
    "PPOLearner": PPO_DEFAULT_CFGS,
    "IMPALALearner": IMPALA_DEFAULT_CFGS,
}
ALL_MATCH_KEYS = {
    "DrQv2Learner": DRQV2_MATCH_KEYS,
    "PPGLearner": PPG_MATCH_KEYS,
    "SACLearner": SAC_MATCH_KEYS,
    "PPOLearner": PPO_MATCH_KEYS,
    "IMPALALearner": IMPALA_MATCH_KEYS,
}
