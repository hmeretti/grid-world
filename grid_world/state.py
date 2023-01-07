from __future__ import annotations
from dataclasses import dataclass

from grid_world.visualization.unicode_definitions import states_symbols
from abstractions import State


@dataclass(frozen=True)
class GWorldState(State):
    """
    State for a grid world
    """

    coordinates: tuple[int, int]
    kind: str = "empty"

    def __add__(self, other: tuple[int, int]) -> tuple[int, int]:
        return self.coordinates[0] + other[1], self.coordinates[1] + other[1]

    def __eq__(self, other):
        return self.coordinates == other.coordinates and self.kind == other.kind

    def get_unicode(self) -> str:
        return states_symbols[self.kind]

    def __str__(self):
        return f"{self.kind} at {self.coordinates}"


@dataclass(frozen=True)
class TagState(State):
    """
    State for the Tag problem
    """

    coordinates_1: tuple[int, int]
    coordinates_2: tuple[int, int]

    def __eq__(self, other):
        return (
            self.coordinates_1 == other.coordinates_1
            and self.coordinates_2 == other.coordinates_2
        )

    def __str__(self):
        return f"agent 1 at {self.coordinates_1}, agent 2 at {self.coordinates_2}"
