class EligibilityTrace:
    def __init__(
        self,
        kind: str = "accumulating",
        et_lambda: float = 0.5,
        gamma: float = 1,
        alpha: float = 1,
    ):
        self.et_dict = {}
        self.et_lambda = et_lambda
        self.gamma = gamma
        self.kind = kind
        self.alpha = alpha

    def __call__(self, *x):
        return self.et_dict.get(x, 0)

    def update(self, *x):
        self.et_dict.update(
            {
                y: self.gamma * self.et_lambda * self.et_dict[y]
                for y in self.et_dict
                if y != x
            }
        )
        self.et_dict[tuple(x)] = self._visited_state_update(x)

    def _visited_state_update(self, x) -> float:
        if self.kind == "replacing":
            return 1
        elif self.kind == "dutch":
            return (1 - self.alpha) * self.gamma * self.et_lambda * self.__call__(
                *x
            ) + 1
        elif self.kind == "accumulating":
            return self.gamma * self.et_lambda * self.et_dict.get(x, 0) + 1
        else:
            raise ValueError("kind must be one of: accumulating, dutch, replacing")

    def get_relevant_states(self):
        return list(self.et_dict.keys())
