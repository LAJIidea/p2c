from p2c.ir.parameter import Parameter
from p2c.ir.stmt import Stmt
from p2c.ir.primitive_type import PrimType
from typing import List


class Func:

    def __init__(self, 
                 name: str, 
                 parameters: List[Parameter], 
                 returned: PrimType,
                 stmts: List[Stmt]) -> None:
        self.name = name
        self.parameters = parameters
        self.returned = returned
        self.stmts = stmts