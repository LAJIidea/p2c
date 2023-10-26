from p2c.kernel.kernel import translate


@translate
def test() -> None:
    a = 1
    print(a)


test.dump()
