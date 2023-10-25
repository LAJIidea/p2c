from p2c.ir.primitive_type import PrimType


class Parameter:
    def __init__(self, name: str, ty: PrimType) -> None:
        self.name = name
        self.ty = ty