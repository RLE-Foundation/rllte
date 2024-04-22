# # load the library
# from rllte.xplore.reward import RE3
# # create the reward module
# irs = RE3(...)
# # reset the environment
# obs, infos = envs.reset()
# # training loop
# while True:
#     # sample actions
#     actions = agent(obs)
#     # step the environment
#     next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
#     # get data from the transitions
#     irs.watch(obs, actions, rewards, next_obs, terminateds, truncateds, infos)
#     # compute the intrinsic rewards at each step
#     ## sync (bool): Whether to update the reward module after the `compute` function, default is `True`
#     intrinsic_rewards = irs.compute(samples=dict(observations=obs, 
#                                                  actions=actions, 
#                                                  rewards=rewards, 
#                                                  terminateds=terminateds, 
#                                                  truncateds=truncateds,
#                                                  next_observations=next_obs), 
#                                     sync=False)
#     ...
#     # update the reward module
#     batch = replay_storage.sample()
#     irs.update(samples=dict(observations=batch.obs, 
#                             actions=batch.actions, 
#                             rewards=batch.rewards, 
#                             terminateds=batch.terminateds, 
#                             truncateds=batch.truncateds,
#                             next_observations=batch.next_obs)
#     )
#     ...


# load the library
from rllte.xplore.reward import RE3
# create the reward module
irs = RE3(...)
# reset the environment
obs, infos = envs.reset()
# a rollout storage
rs = RolloutStorage(...)
# training loop
for episode in range(...):
    for step in range(...):
        # sample actions
        actions = agent(obs)
        # step the environment
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        # get data from the transitions
        irs.watch(obs, actions, rewards, next_obs, terminateds, truncateds, infos)
        ...
    # prepare the samples
    samples = dict(observations=rs.obs, 
                   actions=rs.actions, 
                   rewards=rs.rewards, 
                   terminateds=rs.terminateds, 
                   truncateds=rs.truncateds,
                   next_observations=rs.next_obs
    )
    # compute the intrinsic rewards
    ## sync (bool): Whether to update the reward module after the `compute` function, default is `True`.
    intrinsic_rewards = irs.compute(samples, sync=True)