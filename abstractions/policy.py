from abc import ABC, abstractmethod
from typing import Any

from abstractions import State, Action


class Policy(ABC):
    """
    Abstract policy class.
    """

    @abstractmethod
    def __call__(self, state: State, action: Action) -> float:
        """
        Tells for each state, the probability of an action

        :param state: the state we want to know about
        :param action: the action we want to know about
        """
        raise NotImplementedError("__call__ method not implemented")

    @abstractmethod
    def update(self, *args: Any) -> None:
        """
        A function responsible for updating the policy.

        :param args: different implementations will use different parameters
        """
        raise NotImplementedError("update method not implemented")
