from typing import Iterable, Collection, Collection

import numpy as np

from utils.operations import add_tuples

from grid_world.action import Action
from grid_world.grid_world import GridWorld
from grid_world.state import State
from grid_world.type_aliases import PoliceRec, Police, Q


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
        if (score := q.get((s, a), 0)) > best_score:
            best_score = score
            best_action = a
    return best_action


def get_explorer_police(
    q: Q,
    world_map: set[State],
    actions: Collection[Action],
    reasonable_actions: dict[State, Collection[Action]],
    epsilon: float = 0.1,
):
    """
    This creates ....

    :param q: the Q function
    :param world_map: a set of states, representing a partial map of our world
    :param actions: all possible actions to be considered
    :param reasonable_actions: reasonable  actions worth exploring
    :param epsilon: the parameter that names the function
    :return: our epsilon greedy function
    """
    police_map = {}
    for s in world_map:
        best_action = get_best_action_from_q(q, s, reasonable_actions.get(s, actions))
        p_0 = epsilon / len(reasonable_actions[s])
        for a in actions:
            police_map[s, a] = (
                p_0 + (1 - epsilon if a == best_action else 0)
                if a in reasonable_actions[s]
                else 0
            )

    return (
        lambda cs, ca: police_map[cs, ca]
        if (cs, ca) in police_map.keys()
        else 1 / len(actions)
    )


def get_reasonable_actions(
    world_map: set[State], s: State, actions: Collection[Action]
) -> Collection[Action]:
    no_go_coordinates = [
        cs.coordinates for cs in world_map if cs.kind in {"trap", "wall"}
    ]
    return [
        a
        for a in actions
        if add_tuples(s.coordinates, a.direction) not in no_go_coordinates
    ]
