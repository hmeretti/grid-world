from abc import ABC, abstractmethod
from typing import Any

from abstractions import State, Action


class Policy(ABC):
    """
    Abstract policy class. Concrete extensions should implement

    :__call__: the function that tells for each state, the probability of an action
    :update: a function to update the policy

    """

    @abstractmethod
    def __call__(self, state: State, action: Action) -> float:
        raise NotImplementedError("__call__ method not implemented")

    @abstractmethod
    def update(self, *args: Any) -> None:
        raise NotImplementedError("update method not implemented")
