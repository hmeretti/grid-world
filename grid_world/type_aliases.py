from collections import Callable

from grid_world.action import Action
from grid_world.state import State

Effect = int
Police = Callable[[State, Action], float]
RewardFunction = Callable[[Effect], float]
Q = dict[tuple[State, Action], float]
PoliceRec = dict[State, Action]
EvalFunction = dict[State, float]
