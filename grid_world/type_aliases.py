from collections import Callable

from grid_world.action import Action
from grid_world.state import State

Effect = int
Policy = Callable[[State, Action], float]
RewardFunction = Callable[[Effect], float]
Q = dict[tuple[State, Action], float]
PolicyRec = dict[State, Action]
EvalFunction = dict[State, float]
DecayFunction = Callable[[float], Callable[[float], float]]