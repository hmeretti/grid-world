from typing import TypeVar

from .state import State
from .action import Action
from .type_aliases import (
    EvalFunction,
    WorldModel,
    Effect,
    RewardFunction,
    Q,
    PolicyRec,
    DecayFunction,
)
from .policy import Policy
from .world import World
from .agent import Agent

ActionTypeVar = TypeVar("ActionTypeVar", bound="Action")
StateTypeVar = TypeVar("StateTypeVar", bound="State")
