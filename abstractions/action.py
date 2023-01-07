from abc import ABC
from enum import Enum


class Action(ABC):
    """
    Abstract action class
    """

    pass


# weird shit to get a class to inherit both action and enum
class MetaEnumActionClass(type(Enum), type(Action)):
    pass
