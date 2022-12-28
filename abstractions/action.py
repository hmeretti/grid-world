from abc import ABC
from enum import Enum


class Action(ABC):
    pass


# weird shit to get a class to inherit both action and enum
class MetaEnumActionClass(type(Enum), type(Action)):
    pass
