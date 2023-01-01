from __future__ import annotations
from dataclasses import dataclass

from abstractions import Action


@dataclass(frozen=True)
class TestAction(Action):
    name: str

    def __str__(self):
        return self.name


a0 = TestAction("a0")
a1 = TestAction("a1")
a2 = TestAction("a2")
a3 = TestAction("a3")
