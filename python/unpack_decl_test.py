from p2c.kernel.kernel import translate


@translate
def test():
    a, b, c = 1, 2, 3
    return a + b + c


test.dump()
