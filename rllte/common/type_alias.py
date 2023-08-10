from collections import namedtuple


VanillaReplayBatch = namedtuple(
    typename="VanillaReplayBatch",
    field_names=["observations", "actions", "rewards", "terminateds", "truncateds", "next_observations"],
)

PrioritizedReplayBatch = namedtuple(
    typename="PrioritizedReplayBatch",
    field_names=["observations", "actions", "rewards", "terminateds", "truncateds", "next_observations", "indices", "weights"],
)

NStepReplayBatch = namedtuple(
    typename="NStepReplayBatch", field_names=["observations", "actions", "rewards", "discounts", "next_observations"]
)

VanillaRolloutBatch = namedtuple(
    typename="VanillaRolloutBatch",
    field_names=[
        "observations",
        "actions",
        "values",
        "returns",
        "terminateds",
        "truncateds",
        "old_log_probs",
        "adv_targ",
    ],
)