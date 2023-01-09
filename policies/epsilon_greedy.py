from abstractions import Action, State, DecayFunction, Policy


class EpsilonGreedy(Policy):
    def __init__(
        self,
        epsilon: float = 0.1,
        actions: tuple[Action, ...] = None,
        epsilon_decay: DecayFunction = None,
    ):
        """
        An epsilon greedy policy, which tells the probability of taking an action
        in a state

        :param epsilon: the epsilon parameter
        :param actions: actions available to select from
        :param epsilon_decay: a decaying function that will be applied to epsilon
        whenever the decay method is called
        """
        self.policy_map = {}
        self.best_action = {}
        self._epsilon = epsilon
        self.actions = actions
        self.epsilon_decay = epsilon_decay

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, new_epsilon):
        self._epsilon = new_epsilon
        for state in {x for (x, _) in self.policy_map.keys()}:
            self.update(state, self.best_action[state], force_update=True)

    def __call__(self, state: State, action: Action) -> float:
        if action in self.actions:
            return self.policy_map.get((state, action), 1 / len(self.actions))
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
        if (self.best_action.get(state) != best_action) or force_update:
            self.best_action[state] = best_action
            for cur_a in self.actions:
                self.policy_map[state, cur_a] = (
                    1 - self.epsilon
                    if cur_a == best_action
                    else self.epsilon / (len(self.actions) - 1)
                )

    def decay(self) -> None:
        """
        Decay the value of epsilon and update the policy accordingly
        """
        if self.epsilon_decay is not None:
            self.epsilon = self.epsilon_decay(self.epsilon)
