from p2c.kernel.kernel import translate


@translate
def test(a: int, b: int):
    if a < b:
        return a
    elif a == b:
        return b
    else:
        return 0
    

test.dump()
