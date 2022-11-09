from collections import Callable

State = any
Action = any
EvalFunction = dict[State, float]
Policy = Callable[[State, Action], float]
RewardFunction = Callable[[State, Action], float]
WorldModel = Callable[[State, Action], Callable[State, float]]
