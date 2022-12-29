from abc import ABC, abstractmethod
from typing import Generic

from abstractions import Action, Effect
from abstractions.type_vars import StateTypeVar


class World(ABC, Generic[StateTypeVar]):
    """
    Abstract world class. A world for us is anything that takes an state

    :__call__: the function that tells for each state, the probability of an action
    :update: a function to update the policy

    """

    states: list[StateTypeVar]
    initial_state: StateTypeVar

    @abstractmethod
    def take_action(self, state: StateTypeVar, action: Action) -> [StateTypeVar, Effect]:
        raise NotImplementedError("take_action method not implemented")
