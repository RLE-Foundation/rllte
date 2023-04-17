import hydra # Use Hydra to manage experiments

from hsuanwu.env import make_dmc_env # Import DeepMind Control Suite
from hsuanwu.common.engine import HsuanwuEngine # Import Hsuanwu engine

train_env = make_dmc_env(env_id='cartpole_balance') # Create train env
test_env = make_dmc_env(env_id='cartpole_balance') # Create test env

@hydra.main(version_base=None, config_path='cfgs', config_name='minimum_config')
def main(cfgs):
    engine = HsuanwuEngine(cfgs=cfgs, train_env=train_env, test_env=test_env) # Initialize engine
    engine.invoke() # Start training

if __name__ == '__main__':
    main()