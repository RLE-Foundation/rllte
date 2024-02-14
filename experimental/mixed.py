from rllte.xplore.reward import *

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
    def __init__(self, m1_name, m2_name, m1_kwargs, m2_kwargs):

        m1 = name_to_class[m1_name](**m1_kwargs)
        m2 = name_to_class[m2_name](**m2_kwargs)
        super().__init__(m1, m2)

        assert m1.rwd_norm_type == m2.rwd_norm_type
        assert m1.observation_space == m2.observation_space
        assert m1.action_space == m2.action_space
        assert m1.n_envs == m2.n_envs
        assert m1.device == m2.device
        assert m1.obs_rms == m2.obs_rms
        assert m1.beta == m2.beta
        assert m1.kappa == m2.kappa
        assert m1.rwd_norm_type == m2.rwd_norm_type

        self.rwd_norm_type = m1.rwd_norm_type
        self.observation_space = m1.observation_space
        self.action_space = m1.action_space
        self.n_envs = m1.n_envs
        self.device = m1.device
        self.obs_rms = m1.obs_rms
        self.beta = m1.beta
        self.kappa = m1.kappa
        self.rff = m1.rff
        self.global_step = m1.global_step
    
    def compute(self, samples):
        # get the number of steps and environments
        rewards1, rewards2 = super().compute(samples)

        return rewards1 + rewards2