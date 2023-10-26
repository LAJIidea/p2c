from p2c.kernel.kernel import translate


@translate
def test(a: int):
    a = ~a
    b = not a
    return -a


test.dump()
