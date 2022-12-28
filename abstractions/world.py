from abc import ABC, abstractmethod

from abstractions import State, Action, Effect


class World(ABC):
    """
    Abstract policy class. Concrete extensions should implement

    :__call__: the function that tells for each state, the probability of an action
    :update: a function to update the policy

    """

    states: list[State]

    @property
    @abstractmethod
    def initial_state(self):
        pass

    @abstractmethod
    def take_action(self, state: State, action: Action) -> [State, Effect]:
        raise NotImplementedError("take_action method not implemented")
