from p2c.kernel.kernel import translate
from p2c.kernel.kernel_builder import KernelASTBuilder


@translate
def test1(a: int, b: int):
    return a + b


@translate
def test():
    c = test1(1, 2)
    return c


KernelASTBuilder().build_all()