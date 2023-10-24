from p2c.ast.ast_utils import Builder, ReturnStatus, LoopStatus


class ASTVisitor(Builder):
  
  @staticmethod
  def visitor_Name(ctx, node):
      node.ptr = ctx.get_var_by_name(node.id)
      return node.ptr
  
  @staticmethod
  def visit_AnnAssign(ctx, node):
      visit_stmt(ctx, node.target)

  @staticmethod
  def visitor_Num(ctx, node):
      node.ptr = node.n
      return node.ptr


visit_stmt = ASTVisitor()


def visit_stmts(ctx, stmts):
    with ctx.variable_scope_guard():
        for stmt in stmts:
            if ctx.returned != ReturnStatus.NoReturn or ctx.loop_status() != LoopStatus.Normal:
                break
            else:
                visit_stmt(ctx, stmt)
    return stmts