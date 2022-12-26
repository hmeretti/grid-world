from enum import Enum
from typing import Final


class Action(Enum):
    up = ((0, 1), " \u2191 ")
    down = ((0, -1), " \u2193 ")
    right = ((1, 0), " \u2192 ")
    left = ((-1, 0), " \u2190 ")
    up_right = ((1, 1), " \u2B08 ")
    up_left = ((-1, 1), " \u2B09 ")
    down_right = ((1, -1), " \u2B0A ")
    down_left = ((-1, -1), " \u2B0B ")
    wait = ((0, 0), " \u27F3 ")

    def __init__(self, direction: tuple[int, int], unicode: str):
        self.direction: Final = direction
        self.unicode: Final = unicode
