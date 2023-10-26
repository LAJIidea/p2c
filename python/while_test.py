from p2c.kernel.kernel import translate


@translate
def test(a: int, b: int):
    while a < b:
        a = a + b
    return a


test.dump()
