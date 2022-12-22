from typing import Any

from dynamic_programing.type_aliases import State, Action
from grid_world.grid_world import GridWorld


class Agent:
    """
    Abstract policy class. Concrete extensions should implement

    :__call__: the function that tells for each state, the probability of an action
    :update: a function to update the policy

    """

    def train(
        self,
        world: GridWorld,
        episodes: int,
    ) -> tuple[list[int], list[float]]:
        raise NotImplementedError("train method not implemented")

    def run_episode(
            self, world: GridWorld, initial_state: State
    ) -> tuple[list[float], list[float]]:
        raise NotImplementedError("run_episode method not implemented")
