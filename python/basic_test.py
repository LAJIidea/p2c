from p2c.kernel.kernel import translate


@translate
def sim(a: int, b: int):
    d = [2, 3, 4]
    for i in d:
        if i == a or i == b:
            return i


print(sim)