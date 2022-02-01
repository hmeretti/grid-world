from typing import Iterable, Collection, Collection

import numpy as np

from dynamic_programing.type_aliases import Action
from grid_world.grid_world import GridWorld
from grid_world.type_aliases import PoliceRec, Police, State, Q


def get_police_rec(
    pi: Police, world: GridWorld, actions: Iterable[Action]
) -> PoliceRec:
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


def get_random_police(actions: Collection[Action]) -> Police:
    """
    Builds a random uniform police over a set of actions

    :param actions: Collection of allowed actions over a state, this is
    assumed to be the same for every state
    :return: the random uniform policy
    """
    return lambda s, a: 1 / len(actions)


def sample_action(police: Police, state: State, actions: Collection[Action]) -> Action:
    cum_sum = 0
    n0 = np.random.uniform()
    for action in actions:
        cum_sum += police(state, action)
        if n0 <= cum_sum:
            return action

    raise ValueError("police does not add to 1 over actions")


def get_e_greedy_police(
    q: Q, states: Collection[State], actions: Collection[Action], epsilon: float = 0.1
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
    police_map = {}
    p_0 = epsilon / len(actions)
    for s in states:
        best_action = get_best_action_from_q(q, s, actions)
        for a in actions:
            police_map[s, a] = p_0 + (1 - epsilon if a == best_action else 0)

    return (
        lambda cs, ca: police_map[cs, ca]
        if (cs, ca) in police_map.keys()
        else 1 / len(actions)
    )


def get_best_action_from_q(q: Q, s: State, actions: Collection[Action]) -> Action:
    best_score = float("-inf")
    for a in actions:
        if (score := q[s, a]) > best_score:
            best_score = score
            best_action = a
    return best_action
