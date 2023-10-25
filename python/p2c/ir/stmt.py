from enum import Enum
from typing import List, Dict
from p2c.ir.expr import Expr, ExprType
from p2c.ir.primitive_type import PrimType
from p2c.ir.parameter import Parameter


class Ended(Enum):
    Break = 0,
    Continue = 1,
    Return = 2,
    Pass = 3,


class AugOp(Enum):
    Add = 0,
    Sub = 1,
    Times = 2,
    Div = 3,
    Base = 4,
    Bit = 5,


class StmtType(Enum):
    If = 0,
    For = 1,
    While = 2,
    Decl = 3,
    Assign = 4
    End = 5


class Stmt:
    def __init__(self, stmt_type: StmtType) -> None:
        self.stmt_type = stmt_type

    def isa(self):
        return self.stmt_type


class Decl(Stmt):
    def __init__(self, pairs: Dict[str, Expr], ty: PrimType) -> None:
        self.pairs = pairs
        self.ty = ty
        super().__init__(StmtType.Decl)


class If(Stmt):
    def __init__(self, expr: Expr, body: List[Stmt], elbody: List[Stmt]) -> None:
        self.expr = expr
        self.body = body
        self.elbody = elbody
        super().__init__(StmtType.Decl)


class For(Stmt):
    def __init__(self, init: Parameter, bound: str, body: List[Stmt], size = 0, step = 1) -> None:
        self.init = init
        self.bound = bound
        self.body = body
        self.size = size
        self.step = step
        super().__init__(StmtType.For)


class While(Stmt):
    def __init__(self, expr: Expr, body: List[Stmt]) -> None:
        self.expr = expr
        self.body = body
        super().__init__(StmtType.While)


class Assign(Stmt):
    def __init__(self, left: Expr, right: Expr, op: AugOp) -> None:
        self.left = left
        self.right = right
        self.op = op
        super().__init__(StmtType.Assign)


class End(Stmt):
    def __init__(self, ended: Ended, init = None) -> None:
        self.ended = ended
        self.init = init
        super().__init__(StmtType.End)
