from p2c.ast.ast_utils import Builder


class ASTVisitor(Builder):
  
  @staticmethod
  def build_Name(ctx, node):
      node.ptr = ctx