from typing import Callable

from abstractions import Action
from abstractions import State
from abstractions.type_vars import ActionTypeVar, StateTypeVar

StateEvalDict = dict[State, float]
WorldModel = Callable[[State, Action], Callable[[State], float]]
Effect = int
RewardFunction = Callable[[Effect], float]
StateActionReward = Callable[[State, Action], float]
Q = dict[tuple[State, Action], float]
PolicyRec = dict[StateTypeVar, ActionTypeVar]
