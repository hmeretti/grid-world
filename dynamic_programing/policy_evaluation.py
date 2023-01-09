import numpy as np

from abstractions import (
    State,
    StateEvalDict,
    WorldModel,
    Policy,
    Action,
    StateActionReward,
)


def _acc_v(
    s: State,
    v: StateEvalDict,
    pi: Policy,
    world_model: WorldModel,
    reward_function: StateActionReward,
    actions: tuple[Action, ...],
    states: tuple[State, ...],
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
    reward_function: StateActionReward,
    actions: tuple[Action, ...],
    states: tuple[State, ...],
    v0: StateEvalDict,
    gamma: float = 1,
) -> float:
    v = v0.copy()
    for s in states:
        v0[s] = _acc_v(s, v, pi, world_model, reward_function, actions, states, gamma)
    return np.amax(np.abs([v0[x] - v[x] for x in v0]))


def iterative_policy_evaluation(
    pi: Policy,
    world_model: WorldModel,
    reward_function: StateActionReward,
    actions: tuple[Action, ...],
    states: tuple[State, ...],
    v0: StateEvalDict = None,
    gamma: float = 1,
    epsilon: float = 0.01,
) -> StateEvalDict:
    """
    Function to create evaluation of policy. That is a mapping from states to the estimated
    accumulated discounted reward from that state, when following policy pi.

    :param pi: policy to be evaluated
    :param world_model: dynamics model of the world. A function of states actions, that returns
        the probability distribution of landing in a new state.
    :param reward_function: the reward for taking an action in a given state
    :param actions: all possible actions
    :param states: all possible states
    :param v0: initial policy evaluation to iterate over
    :param gamma: discount factor for rewards
    :param epsilon: stop criteria. Iteration will stop whenever the maximum change on a state
        evaluation is lower than this
    :return: the evaluation of policy pi
    """
    v = {s: 0 for s in states} if v0 is None else v0.copy()

    delta = 2 * epsilon
    while delta > epsilon:
        delta = _iterate_policy_step(
            pi, world_model, reward_function, actions, states, v, gamma
        )

    return v
