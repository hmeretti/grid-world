from abc import ABC, abstractmethod
from typing import Generic

from abstractions import Action, Effect
from abstractions.type_vars import StateTypeVar


class World(ABC, Generic[StateTypeVar]):
    """
    Abstract world class. A world for us is anything that has some states, and can
    receive an action to take one state to another.
    """

    states: list[StateTypeVar]
    initial_state: StateTypeVar

    @abstractmethod
    def take_action(
        self, state: StateTypeVar, action: Action
    ) -> [StateTypeVar, Effect]:
        raise NotImplementedError("take_action method not implemented")
