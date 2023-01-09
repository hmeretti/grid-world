from abstractions import Policy, Action, State


class GreedyPolicy(Policy):
    def __init__(
        self, actions: tuple[Action, ...] = None, policy_map: dict[[State], Action] = None
    ):
        """
        A greedy policy, which tells the probability of taking an action in a state
        as always 1 or 0

        :param actions: actions available to select from
        whenever the decay method is called
        """
        self.policy_map = policy_map if policy_map is not None else {}
        self.actions = actions

    def __call__(self, state: State, action: Action) -> float:
        if action in self.actions:
            return 1 if action == self.policy_map.get(state) else 0
        else:
            raise ValueError(f"action {action} is not part of policy")

    def update(self, state: State, best_action: Action) -> None:
        """
        Updates the best action for a state

        :param state: state to be updated
        :param best_action: the best action for this state
        """
        self.policy_map[state] = best_action
