from rllte.xplore.reward import *

best_config = {
    ""
}

name_to_class = {
    "rnd": RND,
    "pseudo_counts": PseudoCounts,
    "re3": RE3,
    "ride": RIDE,
    "e3b": E3B,
    "disagreement": Disagreement,
    "ngu": NGU,
    "icm": ICM
}

class Mixed(Fabric):
    def __init__(self, m1, m2):
        super().__init__(m1, m2)
        self.rwd_norm_type = m1.rwd_norm_type
        self.n_envs = m1.n_envs
        self.device = m1.device
        self.rff = None
    
    def compute(self, samples):
        # get the number of steps and environments
        rewards1, rewards2 = super().compute(samples)

        return rewards1 + rewards2