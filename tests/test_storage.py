import pytest
import torch as th

from rllte.env import make_bitflipping_env, make_dmc_env, make_minigrid_env
from rllte.xploit.storage import (
    DictReplayStorage,
    DictRolloutStorage,
    HerReplayStorage,
    NStepReplayStorage,
    PrioritizedReplayStorage,
    VanillaReplayStorage,
    VanillaRolloutStorage,
)


@pytest.mark.parametrize(
    "storage_cls",
    [
        NStepReplayStorage,
        PrioritizedReplayStorage,
        # VanillaReplayStorage,
        # VanillaRolloutStorage,
        # DictReplayStorage,
        # DictRolloutStorage,
        # HerReplayStorage
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_storage(storage_cls, device):
    num_envs = 3
    num_steps = 2000
    batch_size = 64

    if storage_cls in [DictRolloutStorage, DictReplayStorage]:
        env = make_minigrid_env(num_envs=num_envs, device=device, fully_numerical=True, fully_observable=False)
    elif storage_cls is HerReplayStorage:
        env = make_bitflipping_env(num_envs=num_envs, device=device)
    else:
        env = make_dmc_env(num_envs=num_envs, device=device)

    if storage_cls in [VanillaRolloutStorage, DictRolloutStorage]:
        storage = storage_cls(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            num_envs=num_envs,
            num_steps=num_steps,
            batch_size=batch_size,
        )
    if storage_cls is HerReplayStorage:
        storage = storage_cls(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            num_envs=num_envs,
            batch_size=batch_size,
            reward_fn=lambda x, y, z: th.rand(size=(int(batch_size * 0.8), 1)),
        )
    else:
        storage = storage_cls(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            num_envs=num_envs,
            batch_size=batch_size,
        )

    obs, infos = env.reset()
    for _ in range(num_steps):
        if storage_cls in [DictRolloutStorage, DictReplayStorage, HerReplayStorage]:
            actions = th.zeros(size=(num_envs,), device=device).long()
        else:
            actions = th.rand(size=(num_envs, env.action_space.shape[0]), device=device)

        next_obs, rews, terms, truncs, infos = env.step(actions)

        if storage_cls in [VanillaRolloutStorage, DictRolloutStorage]:
            log_probs = th.rand(size=(num_envs,), device=device)
            values = th.rand(size=(num_envs,), device=device)
            storage.add(obs, actions, rews, terms, truncs, infos, next_obs, **{"log_probs": log_probs, "values": values})
        else:
            storage.add(obs, actions, rews, terms, truncs, infos, next_obs)

        obs = next_obs

    if storage_cls in [VanillaRolloutStorage, DictRolloutStorage]:
        samples = storage.sample()
        for batch in samples:
            print(batch)
    else:
        storage.sample()

    print("Storage test passed!")
