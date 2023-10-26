from p2c.kernel.kernel import translate


@translate
def test(a: int, b: int):
    c = a + b
    c = a * b
    c = a / b
    c = a - b
    c = a % b
    c = a ** b
    c = a // b
    c = a & b
    c = a | b
    c = a ^ b
    c = a << b
    c = a >> b
    e = 4.0
    f = e + b
    f = e * b
    d = a == b
    d = a != b
    d = a > b
    d = a < b
    d = a >= b
    d = a and b
    d = a or b
    return d


test.dump()
