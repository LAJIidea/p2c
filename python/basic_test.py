from p2c.kernel.kernel import translate


@translate
def outer(a: int, b: int):
    d = a + b
    # @translate
    def inner(c: int):
        return d + c
    return inner


# print(outer(1, 2))
print(outer)