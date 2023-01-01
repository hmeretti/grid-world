from __future__ import annotations
from dataclasses import dataclass

from abstractions import State


@dataclass(frozen=True)
class TestState(State):
    name: str

    def __str__(self):
        return self.name


s0 = TestState("s0")
s1 = TestState("s1")
s2 = TestState("s2")
s3 = TestState("s3")
