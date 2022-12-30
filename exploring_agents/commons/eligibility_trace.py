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

    def __call__(self, *x: any) -> float:
        return self.et_dict.get(x, 0)

    def update(self, *x: any):
        self.et_dict.update(
            {
                y: self.gamma * self.et_lambda * self.et_dict[y]
                for y in self.et_dict
                if y != x
            }
        )
        # self.et_dict[tuple(x)] = self._visited_arguments_update(x)

    def visited_arguments_update(self, *x) -> None:
        if self.kind == "replacing":
            updated_value = 1
        elif self.kind == "dutch":
            updated_value = (1 - self.alpha) * self.gamma * self.et_lambda * self.__call__(
                *x
            ) + 1
        elif self.kind == "accumulating":
            updated_value =  self.gamma * self.et_lambda * self.__call__(*x) + 1
        else:
            raise ValueError("kind must be one of: accumulating, dutch, replacing")
        self.et_dict[tuple(x)] = updated_value

    def get_relevant_arguments(self) -> list[any]:
        return list(self.et_dict.keys())

    def reset(self):
        self.et_dict = {}
