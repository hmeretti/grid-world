from grid_world.action import GWorldAction
from grid_world.state import GWorldState
from abstractions import DecayFunction, Policy


class EpsilonExplorer(Policy):
    def __init__(
        self,
        epsilon: float,
        actions: list[GWorldAction],
        epsilon_decay: DecayFunction = None,
    ):
        """
        Similar to epsilon greedy, except for each state it maintains a set of reasonable
        actions, giving any action outside this set probability 0.

        :param epsilon: the epsilon parameter
        :param actions: actions available to select from
        :param epsilon_decay: a decaying function that will be applied to epsilon
        whenever the decay method is called

        """
        self.policy_map = {}
        self.best_action = {}
        self._epsilon = epsilon
        self.actions = actions
        self.reasonable_actions: dict[[GWorldAction], list[GWorldAction]] = {}
        self.epsilon_decay = epsilon_decay

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, new_epsilon):
        self._epsilon = new_epsilon
        for state in {x for (x, _) in self.policy_map.keys()}:
            self.update(
                state,
                self.best_action[state],
                self.reasonable_actions[state],
                force_update=True,
            )

    def __call__(self, state: GWorldState, action: GWorldAction) -> float:
        valid_actions = self.reasonable_actions.get(state, self.actions)
        if action not in valid_actions:
            return 0
        elif len(valid_actions) == 1:
            return 1
        else:
            return self.policy_map.get((state, action), 1 / len(valid_actions))

    def update(
        self,
        state: GWorldState,
        best_action: GWorldAction,
        valid_actions: list[GWorldAction],
        force_update: bool = False,
    ) -> None:
        """
        Updates probabilities for a state based on what are the actions to be considered for
        this state and what the best action is.

        :param state: state to be updated
        :param best_action: the best action for this state
        :param valid_actions: the set for which actions should be chosen for this state
        :param force_update: whether we should apply update, even if the best
        action didn't change
        """
        if (
            self.best_action.get(state) != best_action
            or self.reasonable_actions.get(state) != valid_actions
            or force_update
        ):
            # only need to update if best action, or valid actions changed for the state
            self.reasonable_actions[state] = valid_actions
            self.best_action[state] = best_action
            for cur_a in valid_actions:
                self.policy_map[state, cur_a] = (
                    1 - self.epsilon
                    if cur_a == best_action
                    else self.epsilon / (len(valid_actions) - 1)
                )

    def decay(self) -> None:
        """
        Decay the value of epsilon and update the policy accordingly
        """
        if self.epsilon_decay is not None:
            self.epsilon = self.epsilon_decay(self.epsilon)
