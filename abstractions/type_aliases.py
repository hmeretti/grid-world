from collections import Callable
from abstractions import Action
from abstractions import State

EvalFunction = dict[State, float]
WorldModel = Callable[[State, Action], Callable[State, float]]
Effect = int
RewardFunction = Callable[Effect, float]
Q = dict[tuple[State, Action], float]
PolicyRec = dict[State, Action]
DecayFunction = Callable[[float], float]
