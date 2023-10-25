from p2c.kernel.kernel import translate


@translate
def sim(a: int, b: int):
  d = 2
  c = a + b
  return c / d


print(sim)
