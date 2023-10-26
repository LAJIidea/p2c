from p2c.kernel.kernel import translate


@translate
def test(a: int, b: int):
    return a + b


test.dump()
