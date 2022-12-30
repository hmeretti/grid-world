from typing import Iterable, Collection

import numpy as np

from abstractions import Action, State, World, PolicyRec, Q, Policy
from abstractions.type_vars import ActionTypeVar
from utils.operations import order_callable


def get_policy_rec(pi: Policy, world: World, actions: Iterable[Action]) -> PolicyRec:
    """
    Generate a function  that tells for each state the action which the policy consider the "best"(the
    one it recommends the agent to take more often)

    :param pi: a policy, i.e. a function that tells for each state action pair, how likely we should take it
    :param world: the world to which this policy applies
    :param actions: available actions to which this policy applies
    :return: a functions that tells for each state the action which the policy consider the "best"(the
    one it recommends the agent to take more often)
    """

    pi_rec = {}
    for s in world.states:
        best_score = 0
        for a in actions:
            if (score := pi(s, a)) > best_score:
                pi_rec[s] = a
                best_score = score

    return pi_rec


def get_random_policy(actions: Collection[Action]):
    """
    Builds a random uniform policy over a set of actions

    :param actions: Collection of allowed actions over a state, this is
    assumed to be the same for every state
    :return: the random uniform policy
    """
    return lambda s, a: 1 / len(actions)


def sample_action(
    policy: Policy, state: State, actions: list[ActionTypeVar]
) -> ActionTypeVar:
    """
    Selects an action to a certain state following a policy.

    :param policy: the policy we will consider
    :param state: a state to select the action from
    :param actions: a list of actions to be considered as options
    :returns: the selected action
    """
    cum_sum = 0
    n0 = np.random.uniform()
    for action in actions:
        cum_sum += policy(state, action)
        if n0 <= cum_sum:
            return action

    raise ValueError(
        f"policy adds to: {cum_sum:.5f} over actions: {actions} in state: {state}"
    )


def sample_action_and_exploration(
    policy: Policy, state: State, actions: list[ActionTypeVar]
) -> [ActionTypeVar, bool]:
    """
    Selects an action to a certain state following a policy.
    Also returns information on weather this was an exploration or not.
    It considers any action different than the most likely as an exploration

    :param policy: the policy we will consider
    :param state: a state to select the action from
    :param actions: a list of actions to be considered as options
    :returns: the selected action, and a flag indicating whether this was the most probable or not

    """
    n0 = np.random.uniform()
    ordered_policy = order_callable(lambda a: policy(state, a), actions)

    cum_sum = 0
    for action, p_value in ordered_policy:
        cum_sum += p_value
        if n0 <= cum_sum:
            return action, n0 > ordered_policy[0][1]

    raise ValueError(
        f"policy adds to: {cum_sum:.5f} over actions: {actions} in state: {state}"
    )


def get_e_greedy_policy(
    q: Q, states: list[State], actions: list[Action], epsilon: float = 0.1
):
    """
    This creates an epsilon greedy policy for a Q function. Where we select the prefered
    action 1 - epsilon times, and the rest we select an action at random uniformly.
    This policy will behave as the uniform random policy for states not listed.

    :param q: the Q function
    :param states: a list of states to make our policy greed at
    :param actions: possible actions to be considered
    :param epsilon: the parameter that names the function
    :return: our epsilon greedy function
    """
    policy_map = {}
    act_len = len(actions)
    p_0 = epsilon / act_len
    for s in states:
        best_action = get_best_action_from_dict(q, s, actions)
        for a in actions:
            policy_map[s, a] = p_0 + (1 - epsilon if a == best_action else 0)

    return lambda cs, ca: policy_map.get((cs, ca), 1 / act_len)


def get_best_action_from_dict(
    q: Q, s: State, actions: list[ActionTypeVar]
) -> ActionTypeVar:
    best_action = actions[0]
    best_score = q.get((s, best_action), 0)
    if len(actions) > 1:
        for a in actions[1:]:
            if (score := q.get((s, a), 0)) > best_score:
                best_score = score
                best_action = a
    return best_action


def get_explorer_policy(
    q: Q,
    world_map: set[State],
    actions: Collection[Action],
    reasonable_actions: dict[State, Collection[Action]],
    epsilon: float = 0.1,
):
    """
    This creates a policy similar to epsilon greedy. However, it uses a partial map of the world, and
    a list of reasonable actions to take at each state to avoid making simple mistakes,
    like hitting walls and falling into traps. The reasonable actions parameter is responsible for deciding
    what the police allows or not.

    :param q: the Q function
    :param world_map: a set of states, representing a partial map of our world
    :param actions: all possible actions to be considered
    :param reasonable_actions: reasonable  actions worth exploring
    :param epsilon: the parameter that names the function
    :return: our epsilon greedy function
    """
    policy_map = {}
    for s in world_map:
        best_action = get_best_action_from_dict(
            q, s, reasonable_actions.get(s, actions)
        )
        p_0 = epsilon / len(reasonable_actions[s])
        for a in actions:
            policy_map[s, a] = (
                p_0 + (1 - epsilon if a == best_action else 0)
                if a in reasonable_actions[s]
                else 0
            )

    return (
        lambda cs, ca: policy_map[cs, ca]
        if (cs, ca) in policy_map.keys()
        else 1 / len(actions)
    )
