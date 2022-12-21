from grid_world.type_aliases import DecayFunction


def get_linear_decay(decay: float = 1, min_value: float = 0) -> DecayFunction:
    return lambda x: max(x - decay, min_value)


def get_exp_decay(
    base: float = 2, decay_lambda: float = 1, min_value: float = 0
) -> DecayFunction:
    return lambda x: max(x * base ** (-decay_lambda), min_value)
