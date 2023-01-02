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

    def update(
        self, state: State, best_action: Action, force_update: bool = False
    ) -> None:
        """
        Updates probabilities for a state based on what the best action for this
        state is

        :param state: state to be updated
        :param best_action: the best action for this state
        :param force_update: whether we should apply update, even if the best
        action didn't change
        """
        pass
