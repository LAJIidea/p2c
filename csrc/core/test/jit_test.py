from p2c.kernel.kernel import translate

@translate
def test(a: int) -> int:
    b = 2
    c = a + b
    return a + 1


test.enable_jit()
test.dumpMLIR()
test.optimizeDCE()
test.dumpMLIR()
test(2)
# test.jit(2)