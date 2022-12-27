from typing import Collection

from abstractions import Action, State, DecayFunction, Policy


class EpsilonGreedy(Policy):
    def __init__(
        self,
        epsilon: float = 0.1,
        actions: Collection[Action] = None,
        epsilon_decay: DecayFunction = None,
    ):
        """
        An epsilon greedy policy, which returns

        :reward_function: the reward function we are trying to maximize
        :actions: actions available to the agent
        """
        self.policy_map = {}
        self.best_action = {}
        self.epsilon = epsilon
        self.actions = actions
        self.epsilon_decay = epsilon_decay

    def __call__(self, state: State, action: Action) -> float:
        return self.policy_map.get((state, action), 1 / len(self.actions))

    def update(
        self, state: State, best_action: Action, force_update: bool = False
    ) -> None:
        if (self.best_action.get(state) != best_action) or force_update:
            self.best_action[state] = best_action
            for cur_a in self.actions:
                self.policy_map[state, cur_a] = (
                    1 - self.epsilon
                    if cur_a == best_action
                    else self.epsilon / (len(self.actions) - 1)
                )

    def decay(self) -> None:
        if self.epsilon_decay is not None:
            self.epsilon = self.epsilon_decay(self.epsilon)
            for state in {x for (x, _) in self.policy_map.keys()}:
                self.update(state, self.best_action[state], force_update=True)
