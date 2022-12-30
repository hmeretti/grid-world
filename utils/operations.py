from typing import Callable, TypeVar

import numpy as np


anyVar = TypeVar("anyVar")


def add_tuples(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def float_dict_comparison(d0: dict[any, float], d1: dict[any, float]) -> bool:
    return np.all([np.isclose(d0[x], d1[x]) for x in d0]) and d0.keys() == d1.keys()


def order_dict(x: dict[[anyVar], float], reverse: bool = True) -> list[tuple[anyVar, float]]:
    return sorted(x.items(), key=lambda item: item[1], reverse=reverse)


def order_callable(
    f: Callable[[anyVar], float], domain: list[any], reverse: bool = True
) -> list[tuple[anyVar, float]]:
    return order_dict({x: f(x) for x in domain}, reverse=reverse)
