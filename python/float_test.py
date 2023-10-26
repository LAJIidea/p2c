from p2c.kernel.kernel import translate


@translate
def test(a: float, b: int):
    c = 4.0
    return a + b + c


test.dump()
