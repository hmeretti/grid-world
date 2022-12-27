from enum import Enum
from typing import Final

from abstractions import Action


class GWorldAction(Enum):
    up: Action = ((0, 1), " \u2191 ")
    down: Action = ((0, -1), " \u2193 ")
    right: Action = ((1, 0), " \u2192 ")
    left: Action = ((-1, 0), " \u2190 ")
    up_right: Action = ((1, 1), " \u2B08 ")
    up_left: Action = ((-1, 1), " \u2B09 ")
    down_right: Action = ((1, -1), " \u2B0A ")
    down_left: Action = ((-1, -1), " \u2B0B ")
    wait: Action = ((0, 0), " \u27F3 ")

    def __init__(self, direction: tuple[int, int], unicode: str):
        self.direction: Final = direction
        self.unicode: Final = unicode
