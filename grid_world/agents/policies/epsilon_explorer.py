from typing import Collection

from grid_world.action import GWorldAction
from grid_world.state import GWorldState
from abstractions import DecayFunction, Policy


class EpsilonExplorer(Policy):
    def __init__(
        self,
        epsilon: float = 0.1,
        actions: Collection[GWorldAction] = tuple(GWorldAction),
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
        self.reasonable_actions = {}
        self.epsilon_decay = epsilon_decay

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
        if self.epsilon_decay is not None:
            self.epsilon = self.epsilon_decay(self.epsilon)
            for state in {x for (x, _) in self.policy_map.keys()}:
                self.update(state, self.best_action[state], force_update=True)
