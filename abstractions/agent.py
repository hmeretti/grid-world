from abc import ABC, abstractmethod

from abstractions import State, Action, Effect, Policy
from utils.policy import sample_action


class Agent(ABC):
    """
    Abstract policy class. Concrete extensions should implement

    :__call__: the function that tells for each state, the probability of an action
    :update: a function to update the policy

    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("init not implemented")

    gamma: float = NotImplemented
    actions: list[Action] = NotImplemented
    policy: Policy = NotImplemented

    def select_action(self, state: State) -> Action:
        return sample_action(self.policy, state, self.actions)

    @abstractmethod
    def run_update(
        self, state: State, action: Action, effect: Effect, next_state: State
    ) -> float:
        raise NotImplementedError("run_update method not implemented")

    @abstractmethod
    def finalize_episode(
        self,
        episode_states: list[State],
        episode_returns: list[float],
        episode_actions: list[Action],
    ):
        raise NotImplementedError("finalize_episode method not implemented")
