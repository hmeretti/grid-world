from abc import ABC, abstractmethod
from typing import Generic

from abstractions import Effect, Policy
from abstractions.type_vars import ActionTypeVar, StateTypeVar


class Agent(ABC, Generic[ActionTypeVar, StateTypeVar]):
    """
    Abstract policy class. Concrete extensions should implement

    :__call__: the function that tells for each state, the probability of an action
    :update: a function to update the policy

    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("init not implemented")

    gamma: float = NotImplemented
    actions: list[ActionTypeVar] = NotImplemented
    policy: Policy = NotImplemented

    @abstractmethod
    def select_action(self, state: StateTypeVar) -> ActionTypeVar:
        """
        selects an action from a state based on the agent policy

        :param state: the state to select the action from
        :return: the selected action
        """
        raise NotImplementedError("select_action method not implemented")

    @abstractmethod
    def run_update(
        self,
        state: StateTypeVar,
        action: ActionTypeVar,
        effect: Effect,
        next_state: StateTypeVar,
    ) -> float:
        raise NotImplementedError("run_update method not implemented")

    @abstractmethod
    def finalize_episode(
        self,
        episode_states: list[StateTypeVar],
        episode_returns: list[float],
        episode_actions: list[ActionTypeVar],
    ):
        raise NotImplementedError("finalize_episode method not implemented")
