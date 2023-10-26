from p2c.kernel.kernel import translate


@translate
def test(a: int, b: int):
    c = [1, 2, 3]
    for i in c:
        if i == a or i == b:
            return i
        

test.dump()
