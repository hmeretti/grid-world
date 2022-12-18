import math
from typing import Callable


def get_linear_decay(
    scale: float = 1, min_value: float = 0
) -> Callable[[float], Callable[[float], float]]:
    return lambda x: lambda y: max(x * (1 - y / scale), min_value)


def get_inv_linear_decay(
    scale: float = 1, min_value: float = 0, offset: int = 0
) -> Callable[[float], Callable[[float], float]]:
    return (
        lambda x: lambda y: max(x / (((y - offset) / scale) + 1), min_value)
        if y > offset
        else x
    )


def get_inv_log_decay(
    base: float = math.e, min_value: float = 0
) -> Callable[[float], Callable[[float], float]]:
    return lambda x: lambda y: max(x / (math.log(y + 1, base) + 1), min_value)


def get_exp_decay(
    base: float = math.e, min_value: float = 0
) -> Callable[[float], Callable[[float], float]]:
    return lambda x: lambda y: max(x * base ** (-y), min_value)
