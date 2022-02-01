from collections import Callable

State = any
Action = any
EvalFunction = dict[State, float]
Police = Callable[[State, Action], float]
RewardFunction = Callable[[State, Action], float]
WorldModel = Callable[[State, Action], Callable[State, float]]
