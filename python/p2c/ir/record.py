from p2c.ast.stack import Stack


class Record:
  log = {}
  closure = {}
  global_var = {}
  closure_scope = Stack()
  kernels = []