from p2c.kernel.kernel import translate
from p2c.kernel.kernel_builder import KernelASTBuilder


@translate
def create_closure(a: int, b: int):
    d = a + b
    def inner(c: int):
        return d + c
    d = inner(3)
    return inner


@translate
def test():
    f = create_closure(1, 2)
    return f(3)


KernelASTBuilder().build_all()
