import hydra

from hsuanwu.env import make_dmc_env
from hsuanwu.common.engine import OffPolicyTrainer

train_env = make_dmc_env(env_id='cartpole_balance')
test_env = make_dmc_env(env_id='cartpole_balance')

@hydra.main(version_base=None, config_path='cfgs', config_name='config')
def main(cfgs):
    trainer = OffPolicyTrainer(train_env=train_env, test_env=test_env, cfgs=cfgs)
    trainer.train()

if __name__ == '__main__':
    main()
