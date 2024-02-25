# a rollout storage for on-policy RL algorithms
rs = RolloutStorage(...)
# prepare the samples
samples = dict(observations=rs.observations[:-1],
               actions=rs.actions,
               rewards=rs.rewards,
               terminates=rs.terminates,
               truncateds=rs.truncateds,
               next_observations=rs.observations[1:]
)
# compute the intrinsic rewards
# the `.update(samples)` will invoked automatically
intrinsic_rewards = irs.compute(samples)

# create the reward module
irs = RE3(...)
# reset the environment
obs, infos = envs.reset()
# training loop
while True:
    # get actions
    actions = agent(obs)
    # environment step
    next_obs, rwds, terms, truncs, infos = envs.step(actions)
    # get data from the transitions
    irs.watch(obs, actions, rwds, terms, truncs, next_obs)
    ...
    