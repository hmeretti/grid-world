import pytest

from abstractions import Q
from tests.constants.actions import a0, a1, a2
from tests.constants.states import s0, s1, s2


@pytest.fixture
def q_0() -> Q:
    return {
        (s0, a0): 5,
        (s0, a1): 4,
        (s0, a2): 3,
        (s1, a0): 5,
        (s2, a2): 5,
    }


@pytest.fixture
def q_1() -> Q:
    return {
        (s0, a0): 23,
    }


@pytest.fixture
def q_2() -> Q:
    return {
        (s0, a0): -10,
        (s0, a1): -5,
    }
