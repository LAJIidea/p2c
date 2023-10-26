from p2c.ir.primitive_type import PrimType
from enum import Enum
from typing import List


class BinaryOp(Enum):
    Add = 0,
    Sub = 1,
    Times = 2,
    Div = 3,
    Floor = 4
    Star = 5,
    BitAnd = 6,
    BitOr = 7,
    BitXor = 8,
    And = 9,
    Or = 10,
    Equal = 11,
    NotEqual = 12,
    Greater = 13,
    Lesser = 14,
    GE = 15,
    LE = 16,
    LShift = 17,
    RShift = 18,
    Mod = 19


class UnaryOp(Enum):
    Minus = 0,
    Tiled = 1,
    Bang = 2


class ExprType(Enum):
    Identifier = 0,
    Num = 1,
    String = 2,
    Bool = 3,
    Call = 4,
    Binary = 5,
    Unary = 6,
    Array = 7,


class Expr:
    def __init__(self, expr_type: ExprType) -> None:
      self.expr_type = expr_type
    
    def isa(self):
       return self.expr_type
    

class Identifier(Expr):
    def __init__(self, name: str, closure=False) -> None:
        self.name = name
        self.closure = closure
        super().__init__(ExprType.Identifier)


class Num(Expr):
    def __init__(self, num) -> None:
        self.num = num
        super().__init__(ExprType.Num)


class String(Expr):
    def __init__(self, content: str) -> None:
        self.content = content
        super().__init__(ExprType.String)


class Boolean(Expr):
    def __init__(self, val: bool) -> None:
        self.val = val
        super().__init__(ExprType.Bool)


class Call(Expr):
    def __init__(self, name: str, args: List[Expr], closure = False) -> None:
        self.name = name
        self.args = args
        self.closure = closure
        super().__init__(ExprType.Call)

  
class Binary(Expr):
    def __init__(self, left: Expr, right: Expr, op: BinaryOp) -> None:
        self.left = left
        self.right = right
        self.op = op
        super().__init__(ExprType.Binary)

  
class Unary(Expr):
    def __init__(self, expr: Expr, op: UnaryOp) -> None:
        self.expr = expr
        self.op = op
        super().__init__(ExprType.Unary)


class Array(Expr):
    def __init__(self, elements: List[Expr]) -> None:
        self.elements = elements
        super().__init__(ExprType.Array)