import ast
from p2c.ast.ast_utils import Builder, ReturnStatus, LoopStatus
from p2c.ir.moduler import Moduler
from p2c.ir.func import Func
from p2c.ir.parameter import Parameter
from p2c.ir.primitive_type import PrimType
from p2c.ir.expr import Expr, Identifier, Num, String, Boolean, Call, Binary, BinaryOp, Unary, UnaryOp
from p2c.ir.stmt import Stmt, If, For, While, Decl, Assign, End, Ended, AugOp
from p2c.ir.record import Record


def var_type_check(ctx, ty, name):
    if ctx.is_var_declared(name):
        if ctx.get_var_by_name(name) is not ty:
            raise Exception("C not support dynamic type")


def builtin_to_primtype(ty):
    if ty is int:
        return PrimType.Int
    if ty is list:
        return PrimType.Array
    if ty is float:
        return PrimType.Float
    if ty is str:
        return PrimType.String
    if ty is bool:
        return PrimType.Bool
    # if ty is function:
    #     return PrimType.Function
    if ty is object:
        return PrimType.Struct
    

def handler_expr_type(expr, ctx):
    if isinstance(expr, Num):
        if type(expr.num) is int:
            return PrimType.Int
        else:
            return PrimType.Float
    if isinstance(expr, String):
        return PrimType.String
    if isinstance(expr, Boolean):
        return PrimType.Bool
    if isinstance(expr, Identifier):
        return ctx.get_var_by_name(expr.name)
    if isinstance(expr, Call):
        if expr.name in Record.log:
            return Record.log[expr.name].returned
        else:
            raise Exception(f"Can not find function {expr.name} declared")
    if isinstance(expr, Binary):
        if expr.op in [BinaryOp.BitAnd, BinaryOp.BitOr, BinaryOp.BitXor, BinaryOp.LShift, BinaryOp.RShift]:
            return PrimType.Int
        if expr.op in [BinaryOp.Add, BinaryOp.Sub, BinaryOp.Div, BinaryOp.Times, BinaryOp.Floor]:
            if handler_expr_type(expr.left, ctx) is PrimType.Int and handler_expr_type(expr.left, ctx) is PrimType.Int:
                return PrimType.Int
            else:
                return PrimType.Float
        if expr.op is BinaryOp.Star:
            if handler_expr_type(expr.left, ctx) is PrimType.Int:
                return PrimType.Int
            else:
                return PrimType.Float
        else:
            return PrimType.Bool
    else:
        if expr.op is UnaryOp.Tiled:
            return PrimType.Int
        if expr.op is UnaryOp.Bang:
            return PrimType.Bool
        else:
            return handler_expr_type(expr.expr, ctx)


class ASTVisitor(Builder):

    @staticmethod
    def visit_Module(ctx, node):
        bodys = []
        with ctx.variable_scope_guard():
            for stmt in node.body:
                bodys.append(visit_stmt(ctx, stmt))
        return Moduler(bodys)

    @staticmethod
    def visit_FunctionDef(ctx, node):
        args = node.args
        parameters = []
        ret = None
        with ctx.variable_scope_guard():
            for i, arg in enumerate(args.args):
                if ctx.func.arguments[i].name == arg.arg:
                    param_ty = builtin_to_primtype(ctx.func.arguments[i].annotation)
                    ctx.create_variable(arg.arg, param_ty)
                    parameters.append(Parameter(arg.arg, param_ty))
                else:
                    ctx.create_variable(arg.arg, None)
                    parameters.append(Parameter(arg.arg, None))
            body = visit_stmts(ctx, node.body)
        if node.returns is not None:
            pass
        else:
            ret = ctx.return_ty
        fn = Func(node.name, parameters, ret, body)
        Record.log[node.name] = fn
        return fn

  
    @staticmethod
    def visit_Name(ctx, node):
        return Identifier(node.id)

    @staticmethod
    def visit_AnnAssign(ctx, node):
        visit_stmt(ctx, node.target)

    @staticmethod
    def visit_Assign(ctx, node):
        inits = visit_stmt(ctx, node.value)
        pairs = {}
        base_ty = None
        if type(inits) is list:
            targets = visit_stmt(ctx, node.targets[0])
            base_ty = handler_expr_type(inits[0], ctx)
            flag = all(handler_expr_type(element, ctx) is base_ty for element in inits)
            if not flag:
                raise Exception("p2c not support multiply type in pack")
            if len(inits) != len(targets):
                raise Exception("Unpacked length is not right")
            for i, init in enumerate(inits):
                var_name = targets[i].name
                if ctx.is_var_declared(var_name):
                    raise Exception("Not support multiply variable assign")
                else:
                    ctx.create_variable(var_name, base_ty)
                pairs[targets[i]] = init
            return Decl(pairs, base_ty)
        else:
            base_ty = handler_expr_type(inits, ctx)
            var_name = node.targets[0].id
            var_type_check(ctx, base_ty, var_name)
            target = Identifier(var_name)
            if not ctx.is_var_declared(var_name):
                ctx.create_variable(var_name, base_ty)
            else:
                if ctx.closure and ctx.is_var_prevs(var_name):
                    target.closure = True
                return Assign(target, AugOp.Base)
            pairs[target] = inits
        return Decl(pairs, base_ty)
        
    @staticmethod
    def visit_Num(ctx, node):
        node.ptr = node.n
        return Num(node.n)
    
    @staticmethod
    def visit_Tuple(ctx, node):
        elements = []
        for element in node.elts:
            elements.append(visit_stmt(ctx, element))
        return elements

    @staticmethod
    def visit_Constant(ctx, node):
        return Num(node.n)
    
    @staticmethod
    def visit_BinOp(ctx, node):
        right = visit_stmt(ctx, node.right)
        left = visit_stmt(ctx, node.left)
        op = {
            ast.Add: BinaryOp.Add,
            ast.Sub: BinaryOp.Sub,
            ast.Mult: BinaryOp.Times,
            ast.Div: BinaryOp.Div,
            ast.FloorDiv: BinaryOp.Floor,
            ast.Mod: BinaryOp.Mod,
            ast.Pow: BinaryOp.Star,
            ast.LShift: BinaryOp.LShift,
            ast.RShift: BinaryOp.RShift,
            ast.BitOr: BinaryOp.BitOr,
            ast.BitAnd: BinaryOp.BitAnd,
            ast.BitXor: BinaryOp.BitXor,
        }.get(type(node.op))
        return Binary(left, right, op)
    
    @staticmethod
    def visit_BoolOp(ctx, node):
        visit_stmts(node.values)
        op = {
            ast.And: BinaryOp.And,
            ast.Or: BinaryOp.Or,
        }.get(type(node.op))
        print(node)

    @staticmethod
    def visit_Compare(ctx, node):
        if isinstance(node.ops[0], ast.In):
            pass
        op = {
            ast.Eq: BinaryOp.Equal,
            ast.NotEq: BinaryOp.NotEqual,
            ast.Lt: BinaryOp.Lesser,
            ast.LtE: BinaryOp.LE,
            ast.Gt: BinaryOp.Greater,
            ast.GtE: BinaryOp.GE
        }.get(type(node.ops[0]))
        left = visit_stmt(ctx, node.left)
        right = visit_stmt(ctx, node.comparators[0])
        return Binary(left, right, op)
    
    @staticmethod
    def visit_Return(ctx, node):
        expr = visit_stmt(ctx, node.value)
        if ctx.return_ty is None:
            ty = handler_expr_type(expr, ctx)
            ctx.return_ty = ty
        return End(Ended.Return, expr)


visit_stmt = ASTVisitor()


def visit_stmts(ctx, stmts):
    body = []
    for stmt in stmts:
        if ctx.returned != ReturnStatus.NoReturn or ctx.loop_status() != LoopStatus.Normal:
            break
        else:
            body.append(visit_stmt(ctx, stmt))
    return body