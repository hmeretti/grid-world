from typing import Collection

from grid_world.action import Action
from grid_world.state import State
from grid_world.type_aliases import Q


def best_q_value(q: Q, state: State, actions: Collection[Action]) -> float:
    ans = float("-inf")
    for a in actions:
        if (qa := q.get((state, a), 0)) > ans:
            ans = qa

    return ans
