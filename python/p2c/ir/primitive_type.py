from enum import Enum


class PrimType(Enum):
    Int = 0,
    Float = 1,
    Bool = 2,
    String = 3,
    Array = 4,
    Struct = 5,
    Function = 6,
    Closure = 7