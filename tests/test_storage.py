import pytest
import torch as th

from rllte.env import make_dmc_env
from rllte.xploit.storage import NStepReplayStorage, PrioritizedReplayStorage, VanillaReplayStorage, VanillaRolloutStorage


@pytest.mark.parametrize(
    "storage_cls", [NStepReplayStorage, PrioritizedReplayStorage, VanillaReplayStorage, VanillaRolloutStorage]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_storage(storage_cls, device):
    env = make_dmc_env(device=device)

    if storage_cls is VanillaRolloutStorage:
        storage = storage_cls(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            batch_size=10,
            num_envs=1,
        )
    else:
        storage = storage_cls(
            observation_space=env.observation_space, action_space=env.action_space, device=device, batch_size=10
        )

    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        action = th.as_tensor(action, dtype=th.float32, device=device).unsqueeze(0)

        next_obs, reward, terminated, truncated, info = env.step(action)

        if storage_cls is VanillaRolloutStorage:
            log_probs = th.rand(size=(1,), device=device)
            values = th.rand(size=(1,), device=device)
            storage.add(obs, action, reward, terminated, truncated, next_obs, log_probs=log_probs, values=values)
        else:
            # TODO: add parallel env support
            storage.add(
                obs[0].cpu().numpy(),
                action[0].cpu().numpy(),
                reward[0].cpu().numpy(),
                terminated[0].cpu().numpy(),
                truncated[0].cpu().numpy(),
                info,
                next_obs[0].cpu().numpy(),
            )

        if terminated or truncated:
            if storage_cls is VanillaRolloutStorage:
                storage.sample()
            else:
                storage.sample(0)
            break

        obs = next_obs

    print("Storage test passed!")
