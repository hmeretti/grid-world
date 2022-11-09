from collections import Callable, Collection

import numpy as np

from dynamic_programing.type_aliases import (
    State,
    EvalFunction,
    Policy,
    RewardFunction,
    WorldModel,
    Action,
)


def _acc_v(
    s: State,
    v: EvalFunction,
    pi: Policy,
    world_model: WorldModel,
    reward_function: RewardFunction,
    actions: Collection[Action],
    states: Collection[State],
    gamma: float,
) -> float:
    return sum(
        [
            pi(s, a)
            * sum(
                [
                    world_model(s, a)(s0) * (reward_function(s, a) + gamma * v[s0])
                    for s0 in states
                ]
            )
            for a in actions
        ]
    )


def _iterate_policy_step(
    pi: Policy,
    world_model: WorldModel,
    reward_function: RewardFunction,
    actions: Collection[Action],
    states: Collection[State],
    v0: EvalFunction,
    gamma: float = 1,
) -> float:
    v = v0.copy()
    for s in states:
        v0[s] = _acc_v(s, v, pi, world_model, reward_function, actions, states, gamma)
    return np.amax(np.abs([v0[x] - v[x] for x in v0]))


def iterative_policy_evaluation(
    pi: Policy,
    world_model: WorldModel,
    reward_function: RewardFunction,
    actions: Collection[Action],
    states: Collection[State],
    v0: EvalFunction = None,
    gamma: float = 1,
    epsilon: float = 0.01,
) -> EvalFunction:
    """
    Function to create evaluation of policy. That is a mapping from states to the estimated
    accumulated discounted reward from that state, when following policy pi.

    :param pi: policy to be evaluated
    :param world_model: dynamics model of the world. A function of states actions, that returns
        the probability distribution of landing in a new state.
    :param reward_function: the reward for taking an action in a given state
    :param actions: all possible actions
    :param states: all possible states
    :param v0: initial policy to iterate over
    :param gamma: discount factor for rewards
    :param epsilon: stop criteria. Iteration will stop whenever the maximum change on a state
        evaluation is lower then this
    :return: the evaluation of policy pi
    """
    v = {a: 0 for a in states} if v0 is None else v0.copy()

    delta = 2 * epsilon
    while delta > epsilon:
        delta = _iterate_policy_step(
            pi, world_model, reward_function, actions, states, v, gamma
        )

    return v
