from typing import Any

from dynamic_programing.type_aliases import State, Action


class Policy:
    """
    Abstract policy class. Concrete extensions should implement

    :__call__: the function that tells for each state, the probability of an action
    :update: a function to update the policy

    """

    def __call__(self, state: State, action: Action) -> float:
        raise NotImplementedError("Call magic method not implemented")

    def update(self, *args: Any) -> None:
        raise NotImplementedError("Update method not implemented")
