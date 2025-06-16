from enum import Enum


class TargetType(Enum):
    OPPONENT = 0
    ALLY = 1


class Action(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    SHOOT = 3
