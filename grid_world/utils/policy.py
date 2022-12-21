from typing import Iterable, Collection

import numpy as np

from grid_world.action import Action
from grid_world.grid_world import GridWorld
from grid_world.state import State
from grid_world.type_aliases import PolicyRec, Policy, Q


def get_policy_rec(
    pi: Policy, world: GridWorld, actions: Iterable[Action]
) -> PolicyRec:
    """
    Generate a function  that tells for each state the action which the policy.py consider the "best"(the
    one it recommends the agent to take more often)

    :param pi: a policy.py, i.e. a function that tells for each state action pair, how likely we should take it
    :param world: the world to which this policy.py applies
    :param actions: available actions to which this policy.py applies
    :return: a functions that tells for each state the action which the policy.py consider the "best"(the
    one it recommends the agent to take more often)
    """
    if actions is None:
        actions = Action

    pi_rec = {}
    for s in world.states:
        best_score = 0
        for a in actions:
            if (score := pi(s, a)) > best_score:
                pi_rec[s] = a
                best_score = score

    return pi_rec


def get_random_policy(actions: Collection[Action]) -> Policy:
    """
    Builds a random uniform policy.py over a set of actions

    :param actions: Collection of allowed actions over a state, this is
    assumed to be the same for every state
    :return: the random uniform policy.py
    """
    return lambda s, a: 1 / len(actions)


def sample_action(policy: Policy, state: State, actions: Collection[Action]) -> Action:
    cum_sum = 0
    n0 = np.random.uniform()
    for action in actions:
        cum_sum += policy(state, action)
        if n0 <= cum_sum:
            return action

    raise ValueError("policy.py does not add to 1 over actions")


def get_e_greedy_policy(
    q: Q, states: Collection[State], actions: Collection[Action], epsilon: float = 0.1
):
    """
    This creates an epsilon greedy policy.py for a Q function. Where we select the prefered
    action 1 - epsilon times, and the rest we select an action at random uniformly.
    This policy.py will behave as the uniform random policy.py for states not listed.

    :param q: the Q function
    :param states: a list of states to make our policy.py greed at
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


def get_best_action_from_dict(q: Q, s: State, actions: Collection[Action]) -> Action:
    best_score = float("-inf")
    for a in actions:
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
    This creates a policy.py similar to epsilon greedy. However, it uses a partial map of the world, and
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
        best_action = get_best_action_from_dict(q, s, reasonable_actions.get(s, actions))
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
