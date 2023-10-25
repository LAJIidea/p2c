from p2c.ast.ast_utils import ASTContext
from p2c.ast.ast_visitor import ASTVisitor


def transform_ast(tree, ctx: ASTContext) -> any:
    ir = ASTVisitor()(ctx, tree)
    return ir