from abc import ABC, abstractmethod


class DecayFunction(ABC):
    """
    Abstract decaying function callable.
    """

    @abstractmethod
    def __call__(self, x: float) -> float:
        raise NotImplementedError("__call__ method not implemented")
