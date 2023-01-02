from typing import Final, Collection, Callable

from grid_world.action import GWorldAction
from grid_world.state import GWorldState
from abstractions import Effect

from utils.operations import add_tuples


class GridWorld:
    def __init__(
        self,
        grid_shape: tuple[int, int],
        terminal_states_coordinates: Collection[tuple[int, int]] = None,
        initial_state_coordinates: tuple[int, int] = (0, 0),
        initial_state_coordinates_2: tuple[int, int] = None,
        walls_coordinates: Collection[tuple[int, int]] = None,
        traps_coordinates: Collection[tuple[int, int]] = None,
        wind: Callable[[GWorldState], GWorldAction] = None,
    ):
        """
        This is a class representing a grid world with some simple features.

        States are represented by a tuple of integers, indicating the coordinates in the world and a kind
        indicating if its empty, a trap, a wall, etc...

        An agent can take an action at a state, this will result in a new state and an effect.
        The effect is 0 except when the agent is in a special states. This is just a signal from the
        world to the agent indicating what happened, how the agent will interpret it (i.e. the reward)
        is up to him.

        The legal actions in the world are: up(u), down(d), left(l), right(r), up right(ur),
        up left(ul), down right(dr), down left(dl) and wait(w).
        Whether the agent can take all of then or just a few is up to him. However this a restriction
        on the actions that can be taken(for instance the agent can't move two to the right at once), this is
        done to simplify the dynamics.

        :param grid_shape: The shape of the world
        :param terminal_states_coordinates: coordinates for terminal states, actions in these return an effect of +1
            and won't move the agent
        :param initial_state_coordinates: coordinates for a initial state
        :param walls_coordinates: coordinates for walls, these will be blocked so the agent can't enter then;
            walls are not valid states
        :param traps_coordinates: coordinates for traps. These are valid states, however taking action in then will
            return an effect of -1 and redirect to the initial state
        :param wind: function that for each state returns an action(like the wind pushing the agent in a direction).
            Wind applies a movement after the original movement from the user action based on the landing square.

        """
        self.grid_size: Final[int] = grid_shape[0] * grid_shape[1]
        self.grid_shape: Final = grid_shape
        self.initial_state: Final[GWorldState] = GWorldState(
            initial_state_coordinates, "initial"
        )
        self.initial_state_2: Final[GWorldState] = GWorldState(
            initial_state_coordinates_2, "empty"
        )
        self.walls_coordinates: Final[Collection[GWorldState]] = (
            walls_coordinates if walls_coordinates else tuple()
        )
        self.traps_coordinates: Final[Collection[GWorldState]] = (
            traps_coordinates if traps_coordinates else tuple()
        )
        self.wind: Final[Callable[[GWorldState], GWorldAction]] = wind
        self.terminal_states_coordinates: Final = (
            terminal_states_coordinates
            if terminal_states_coordinates is not None
            else []
        )
        self.states: Final[tuple[GWorldState, ...]] = tuple(
            [
                self._coordinates_to_state((i, j))
                for i in range(grid_shape[0])
                for j in range(grid_shape[1])
                if (i, j) not in self.walls_coordinates
            ]
        )
        self.state_effect: Final[dict[GWorldState, int]] = self._get_state_effect()

    def _get_state_effect(self) -> dict[GWorldState, int]:
        state_effect = {}
        for s in self.states:
            if s.kind == "terminal":
                state_effect[s] = 1
            elif s.kind == "trap":
                state_effect[s] = -1
            else:
                state_effect[s] = 0

        return state_effect

    def take_action(
        self, state: GWorldState, action: GWorldAction
    ) -> [GWorldState, Effect]:
        """
        Represents the effect of an agent taking an action in the world

        :param state: the state the agent is when taking the action
        :param action: the action being taken
        :return: a tuple indicating the resulting state and an integer indicating the effect of the action:
            0: normal transition
            1: agent is in a terminal state
            -1: agent is in a 'trap' state
        """
        if state.kind == "terminal":
            # if we are in a terminal state nothing happens
            pass

        elif state.kind == "trap":
            # if we are in a trap we get sent to the starting position
            state = self.initial_state

        else:
            # else we make the move
            state = self._apply_action(state, action)
            # then apply the wind effect
            if self.wind is not None:
                wind_action = self.wind(state)
                state = self._apply_action(state, wind_action)

        # effect depends only on the state we ended at
        return state, self.state_effect[state]

    def get_state(self, coordinates: tuple[int, int]) -> GWorldState:
        """
        Gets a state from some coordinates.
        """
        try:
            return next(x for x in self.states if x.coordinates == coordinates)
        except StopIteration:
            raise KeyError(f"{coordinates} does not correspond to a valid state")

    def _apply_action(self, state: GWorldState, action: GWorldAction) -> GWorldState:
        coordinates = add_tuples(state.coordinates, action.direction)
        try:
            fs = self._coordinates_to_state(coordinates)
        except KeyError:
            fs = state

        return fs

    def _walkable_state(self, state: GWorldState) -> bool:
        return (
            (0 <= state.coordinates[0] < self.grid_shape[0])
            and (0 <= state.coordinates[1] < self.grid_shape[1])
            and state.kind != "wall"
        )

    def _coordinates_to_state(self, coordinates: tuple[int, int]) -> GWorldState:
        if (
            (0 <= coordinates[0] < self.grid_shape[0])
            and (0 <= coordinates[1] < self.grid_shape[1])
            and (kind := self._coordinates_kind(coordinates)) != "wall"
        ):
            return GWorldState(coordinates, kind)
        else:
            raise KeyError(f"{coordinates} does not correspond to a valid state")

    def _coordinates_kind(self, coordinates: tuple[int, int]) -> str:
        if coordinates in self.terminal_states_coordinates:
            return "terminal"
        elif coordinates in self.walls_coordinates:
            return "wall"
        elif coordinates in self.traps_coordinates:
            return "trap"
        elif coordinates == self.initial_state.coordinates:
            return "initial"
        else:
            return "empty"
