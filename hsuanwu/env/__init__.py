try:
    from .atari import make_atari_env as make_atari_env
    from .dmc import make_dmc_env as make_dmc_env
    from .minigrid import make_minigrid_env as make_minigrid_env
    from .procgen import make_procgen_env as make_procgen_env
except:
    pass
