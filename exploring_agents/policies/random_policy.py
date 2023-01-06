from abstractions import Action, State, Policy


class RandomPolicy(Policy):
    def __init__(
        self,
        actions: list[Action] = None,
    ):
        """
        A random(uniform) policy over a set of actions

        :param actions: actions available to select from
        """
        self.actions = actions

    def __call__(self, state: State, action: Action) -> float:
        if action in self.actions:
            return 1 / len(self.actions)
        else:
            raise ValueError(f"action {action} is not part of policy")

    def update(self) -> None:
        """
        Do nothing at all
        """
        pass
