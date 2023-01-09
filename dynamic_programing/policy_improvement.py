from typing import Collection

from dynamic_programing.policy_evaluation import iterative_policy_evaluation
from abstractions import (
    StateEvalDict,
    WorldModel,
    State,
    Action,
    Policy,
    StateActionReward,
)
from policies import GreedyPolicy, RandomPolicy

from utils.operations import float_dict_comparison


def q_estimate(
    s: State,
    a: Action,
    v: StateEvalDict,
    world_model: WorldModel,
    reward_function: StateActionReward,
    states: Collection[State],
) -> float:
    """
    This is an estimate of the q function for pi bootstrapping from V.
    """
    return reward_function(s, a) + sum([world_model(s, a)(s0) * v[s0] for s0 in states])


def _argmax_q(
    s: State,
    v: StateEvalDict,
    world_model: WorldModel,
    reward_function: StateActionReward,
    actions: tuple[Action, ...],
    states: tuple[State, ...],
) -> Action:
    best_action = actions[0]
    best_score = q_estimate(s, best_action, v, world_model, reward_function, states)
    for a in actions[1:]:
        qa = q_estimate(s, a, v, world_model, reward_function, states)
        if qa > best_score:
            best_score = qa
            best_action = a

    return best_action


def get_greedy_policy(
    v: StateEvalDict,
    world_model: WorldModel,
    reward_function: StateActionReward,
    actions: tuple[Action, ...],
    states: tuple[State, ...],
) -> Policy:
    """
    Creates a greed policy with respect to an evaluation function. Since this function only
    takes into account states and not actions(its the V not the q from the literature), we
    still need the world model and reward function to create this policy.

    :param v: the evaluation function to "greedify" over
    :param world_model: dynamics model of the world. A function of states actions, that returns
        the probability distribution of landing in a new state.
    :param reward_function: the reward for taking an action in a given state
    :param actions: all possible action
    :param states: all possible states
    :return: the greedy policy
    """
    gpr = {
        s: _argmax_q(s, v, world_model, reward_function, actions, states)
        for s in states
    }

    return GreedyPolicy(actions, gpr)


def _dpi_step(
    v_pi, world_model, reward_function, actions, states
) -> [Policy, StateEvalDict]:
    pi_1 = get_greedy_policy(v_pi, world_model, reward_function, actions, states)
    v_pi_1 = iterative_policy_evaluation(
        pi_1, world_model, reward_function, actions, states, v_pi
    )

    return pi_1, v_pi_1


def dynamic_programing_gpi(
    world_model: WorldModel,
    reward_function: StateActionReward,
    actions: tuple[Action, ...],
    states: tuple[State, ...],
    pi: Policy = None,
    v0: StateEvalDict = None,
    max_epochs: int = 100,
) -> [Policy, StateEvalDict]:
    """
    General policy improvement algorithm using dynamic programing.

    :param world_model: dynamics model of the world. A function of states actions, that returns
        the probability distribution of landing in a new state.
    :param reward_function: the reward for taking an action in a given state
    :param actions: all possible actions
    :param states: all possible states
    :param pi: an initial policy, will use random uniform if none is passed
    :param v0: estimated EvalFunction for pi, can speed things up
    :param max_epochs: max number of policy iterations to run(will stop early if the value
        function converges)
    :return: the optimal policy and its value function
    """

    if pi is None:
        pi = RandomPolicy(actions)

    v_pi = iterative_policy_evaluation(
        pi, world_model, reward_function, actions, states, v0
    )

    for i in range(max_epochs):
        v_pi_0 = v_pi
        pi, v_pi = _dpi_step(v_pi, world_model, reward_function, actions, states)

        if float_dict_comparison(v_pi, v_pi_0):
            break

    return pi, v_pi
