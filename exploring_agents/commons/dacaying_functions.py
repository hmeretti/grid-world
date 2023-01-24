from abstractions import DecayFunction


class LinearDecay(DecayFunction):
    def __init__(self, decay: float = 1, min_value: float = 0):
        self.decay = decay
        self.min_value = min_value

    def __call__(self, x: float) -> float:
        return max(x - self.decay, self.min_value)


class ExpDecay(DecayFunction):
    def __init__(self, base: float = 2, decay_lambda: float = 1, min_value: float = 0):
        self.base = base
        self.decay_lambda = decay_lambda
        self.min_value = min_value

    def __call__(self, x: float) -> float:
        return max(x * self.base ** (-self.decay_lambda), self.min_value)
