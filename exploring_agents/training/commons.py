import random

from grid_world.grid_world import GridWorld
from grid_world.state import GWorldState


def select_random_state(
    world: GridWorld, invalid_states: list[GWorldState] = None
) -> GWorldState:
    """
    Selects a random empty, initial or terminal state from a grid world

    :param world: the world we will choose from
    :param invalid_states: states we may want to ignore
    :return: a random state
    """
    invalid_states = [] if invalid_states is None else invalid_states
    return random.choice(
        [
            s
            for s in world.states
            if (s not in invalid_states and s.kind not in ["trap", "wall"])
        ]
    )
