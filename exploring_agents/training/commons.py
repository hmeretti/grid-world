import random

from grid_world.grid_world import GridWorld
from grid_world.state import GWorldState


def select_random_state(
    world: GridWorld, invalid_states: list[GWorldState] = None
) -> GWorldState:
    invalid_states = [] if invalid_states is None else invalid_states
    return random.choice(
        [
            s
            for s in world.states
            if (s not in invalid_states and s.kind not in ["trap", "wall"])
        ]
    )
