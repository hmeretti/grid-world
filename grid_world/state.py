from __future__ import annotations
from dataclasses import dataclass

from grid_world.visualization.unicode_definitions import states_symbols
from abstractions import State


@dataclass(frozen=True)
class GWorldState(State):
    coordinates: tuple[int, int]
    kind: str = "empty"

    def __add__(self, other: tuple[int, int]) -> tuple[int, int]:
        return self.coordinates[0] + other[1], self.coordinates[1] + other[1]

    def get_unicode(self) -> str:
        return states_symbols[self.kind]
