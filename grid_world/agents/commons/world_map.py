from typing import Collection, Final

from grid_world.state import State
from grid_world.action import Action

from utils.operations import add_tuples


class WorldMap:
    def __init__(
        self,
        world_states: set[State] = None,
        actions: Collection[Action] = None,
    ):
        self.world_states: set[State] = (
            world_states if world_states is not None else set()
        )
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.no_go_coordinates: list[tuple[int, ...]] = self._get_no_go_coordinates()
        self.reasonable_actions: dict[
            State, list[Action]
        ] = self._get_reasonable_actions()

    def update_map(self, state: State, action: Action, new_state: State) -> bool:
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
                State(add_tuples(state.coordinates, action.direction), "wall")
            )
            meaningful_update = True
        else:
            self.world_states.add(new_state)
        self.no_go_coordinates = self._get_no_go_coordinates()
        self.reasonable_actions = self._get_reasonable_actions()
        return True if new_state.kind == "trap" else meaningful_update

    def _get_no_go_coordinates(self) -> list[tuple[int, ...]]:
        return [s.coordinates for s in self.world_states if s.kind in {"trap", "wall"}]

    def _get_reasonable_actions(self) -> dict[State, list[Action]]:
        return {s: self._get_actions_for_state(s) for s in self.world_states}

    def _get_actions_for_state(self, s: State) -> list[Action]:
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
