from typing import Final

from grid_world.state import GWorldState
from grid_world.action import GWorldAction

from utils.operations import add_tuples


class WorldMap:
    def __init__(
        self,
        actions: tuple[GWorldAction],
        world_states: set[GWorldState] = None,
    ):
        self.world_states: set[GWorldState] = (
            world_states if world_states is not None else set()
        )
        self.actions: Final = actions
        self.no_go_coordinates: list[tuple[int, int]] = self._get_no_go_coordinates()
        self.reasonable_actions: dict[
            GWorldState, list[GWorldAction]
        ] = self._get_reasonable_actions()

    def update_map(
        self, state: GWorldState, action: GWorldAction, new_state: GWorldState
    ) -> bool:
        """
        Updates map based given a State, Action, State sequence.
        :param state: the original state
        :param action: the action taken
        :param new_state: the final state

        :return: flag indicating whether we added a trap or wall to the map
        """
        meaningful_update = False
        if (new_state == state) and (state.kind != "terminal"):
            self.world_states.add(
                GWorldState(add_tuples(state.coordinates, action.direction), "wall")
            )
            meaningful_update = True
        else:
            self.world_states.add(new_state)
        self.no_go_coordinates = self._get_no_go_coordinates()
        self.reasonable_actions = self._get_reasonable_actions()
        return True if new_state.kind == "trap" else meaningful_update

    def _get_no_go_coordinates(self) -> list[tuple[int, int]]:
        return [s.coordinates for s in self.world_states if s.kind in {"trap", "wall"}]

    def _get_reasonable_actions(self) -> dict[GWorldState, list[GWorldAction]]:
        return {s: self._get_actions_for_state(s) for s in self.world_states}

    def _get_actions_for_state(self, s: GWorldState) -> list[GWorldAction]:
        """
        Generate a list of "reasonable" actions for a specific state. This is done by using a partial
        map of our world to filter out actions which would lead to undesirable results.

        :param s: the state to generate actions for
        :return: list of reasonable actions to take on that state
        """
        return [
            a
            for a in self.actions
            if add_tuples(s.coordinates, a.direction) not in self.no_go_coordinates
        ]
