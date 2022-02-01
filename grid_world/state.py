from __future__ import annotations
from dataclasses import dataclass

from grid_world.visualization.unicode_definitions import states_symbols


@dataclass(frozen=True)
class State:
    coordinates: tuple[int, ...]
    kind: str = "empty"

    def __add__(self, other: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(p + q for p, q in zip(self.coordinates, other))

    def get_unicode(self) -> str:
        return states_symbols[self.kind]
