from p2c.kernel.kernel import translate


@translate
def test():
    a = [1, 2, 3]
    print(a[0])


test.dump()
