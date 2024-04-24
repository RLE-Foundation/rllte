# load the library
from rllte.xplore.reward import RE3
# create the reward module
irs = RE3(...)
# reset the environment
obs, infos = envs.reset()
# training loop
while True:
    # sample actions
    actions = agent(obs)
    # step the environment
    next_obs, rwds, terms, truncs, infos = envs.step(actions)
    # get data from the transitions
    irs.watch(obs, actions, rwds, next_obs, terms, truncs, infos)
    # compute the intrinsic rewards at each step
    ## sync (bool): Whether to update the reward module after the 
    # `compute` function, default is `True`
    intrinsic_rewards = irs.compute(
        samples=dict(observations=obs, actions=actions, 
                     rewards=rwds, terminateds=terms, 
                     truncateds=terms, next_observations=next_obs), 
        sync=False)
    ...
    # update the reward module
    batch = replay_storage.sample()
    irs.update(samples=dict(observations=batch.obs, 
                            actions=batch.actions, 
                            rewards=batch.rewards, 
                            terminateds=batch.terminateds, 
                            truncateds=batch.truncateds,
                            next_observations=batch.next_obs)
    )
    ...