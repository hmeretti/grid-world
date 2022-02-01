import numpy as np


def add_tuples(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(p + q for p, q in zip(a, b))


def float_dict_comparison(d0: dict[any, float], d1: dict[any, float]) -> bool:
    return np.all([np.isclose(d0[x], d1[x]) for x in d0]) and d0.keys() == d1.keys()
