import ast
from p2c.ast.ast_utils import Builder, ReturnStatus, LoopStatus
from p2c.ir.moduler import Moduler
from p2c.ir.func import Func
from p2c.ir.parameter import Parameter
from p2c.ir.primitive_type import PrimType
from p2c.ir.expr import Expr, Identifier, Num, String, Boolean, Call, Binary, BinaryOp, Unary, UnaryOp, Array
from p2c.ir.stmt import Stmt, If, For, While, Decl, Assign, End, Ended, AugOp, Closure
from p2c.ir.record import Record


def var_type_check(ctx, ty, name):
    if ctx.is_var_declared(name):
        if ctx.get_var_by_name(name) is not ty:
            raise Exception("C not support dynamic type")
        

def catch_closure_var(var_name, base_ty):
    var = Parameter(var_name, base_ty)
    if var not in Record.closure_scope.peek():
        Record.closure_scope.peek().append(Parameter(var_name, base_ty))


def type_str_to_primtype(ty):
    if ty == "int":
        return PrimType.Int
    if ty == "float":
        return PrimType.Float
    if ty == "List":
        return PrimType.Array
    if ty == "bool":
        return PrimType.Bool
    if ty == "str":
        return PrimType.String
    if ty == "closure":
        return PrimType.Closure


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
    if isinstance(expr, Array):
        return handler_expr_type(expr.elements[0], ctx)
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
            ctx.create_variable("new_closure", PrimType.Closure)
            for stmt in node.body:
                bodys.append(visit_stmt(ctx, stmt))
        return Moduler(bodys)

    @staticmethod
    def visit_FunctionDef(ctx, node):
        # Just support one nested function
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
                    param_ty = type_str_to_primtype(arg.annotation.id)
                    ctx.create_variable(arg.arg, param_ty)
                    parameters.append(Parameter(arg.arg, param_ty))
            if ctx.inner_func > 0 and ctx.closure is not True:
                ctx.closure = True
                Record.closure_scope.push([])
            ctx.inner_func += 1
            body = visit_stmts(ctx, node.body)
        if node.returns is not None:
            pass
        else:
            ret = ctx.return_ty
        fn = Func(node.name, parameters, ret, body)
        ctx.return_ty = None
        Record.log[node.name] = fn
        ctx.inner_func -= 1
        flag = ctx.closure
        if flag and ctx.inner_func <= 1:
            ctx.closure = False
        if flag:
            closure_data = Record.closure_scope.peek()
            closure = Closure(fn, closure_data)
            Record.closure[node.name] = closure
            Record.closure_scope.pop()
            ctx.create_variable(node.name, PrimType.Closure)
            Record.log["new_closure"] = fn
            return closure
        return fn

  
    @staticmethod
    def visit_Name(ctx, node):
        if ctx.current_scope().get(node.id) is not None:
            if ctx.current_scope()[node.id] is PrimType.Closure:
                return Identifier("new_closure") 
        id = Identifier(node.id)
        if ctx.closure and ctx.is_var_prevs(node.id, 1):
            id.closure = True
            catch_closure_var(node.id, ctx.get_var_by_name(node.id))
        return id

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
            var_name = node.targets[0].id
            if isinstance(inits, Array):
                ctx.list_data[var_name] = inits
            base_ty = handler_expr_type(inits, ctx)
            var_type_check(ctx, base_ty, var_name)
            target = Identifier(var_name)
            if not ctx.is_var_declared(var_name):
                ctx.create_variable(var_name, base_ty)
            else:
                if ctx.closure and ctx.is_var_prevs(var_name, 1):
                    target.closure = True
                    catch_closure_var(var_name, base_ty)
                return Assign(target, inits, AugOp.Base)
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
        values = visit_stmts(ctx, node.values)
        op = {
            ast.And: BinaryOp.And,
            ast.Or: BinaryOp.Or,
        }.get(type(node.op))
        return Binary(values[0], values[1], op)

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
    
    @staticmethod
    def visit_If(ctx, node):
        with ctx.variable_scope_guard():
            expr = visit_stmt(ctx, node.test)
            # some issue here, `if a:` is right, but a is not boolean 
            if handler_expr_type(expr, ctx) is not PrimType.Bool:
                raise Exception("If condition must be bool")
            body = visit_stmts(ctx, node.body)
            orelse = visit_stmts(ctx, node.orelse)
        return If(expr, body, orelse)
    
    @staticmethod
    def visit_List(ctx, node):
        elements = visit_stmts(ctx, node.elts)
        if len(elements) == 0:
            raise Exception("Array init must have elements")
        ty = handler_expr_type(elements[0], ctx)
        flag = all(handler_expr_type(element, ctx) is ty for element in elements)
        if not flag:
            raise Exception("p2c not support multiply type in array")
        return Array(elements)
    
    @staticmethod
    def visit_For(ctx, node):
        with ctx.variable_scope_guard():
            # Just support list iterations
            # !!! Notice: lower is can not be a variable name
            if node.iter.id not in ctx.list_data:
                raise Exception(f"Can not found array {node.iter.id}")
            size = len(ctx.list_data[node.iter.id].elements)
            ty = handler_expr_type(ctx.list_data[node.iter.id], ctx)
            init = Parameter(node.target.id, ty)
            ctx.create_variable(node.target.id, ty)
            bound = node.iter.id
            body = visit_stmts(ctx, node.body)
        return For(init, bound, body, size, 1)
    
    @staticmethod
    def visit_While(ctx, node):
        with ctx.variable_scope_guard():
            expr = visit_stmt(ctx, node.test)
            body = visit_stmts(ctx, node.body)
        return While(expr, body)
    
    @staticmethod
    def visit_Call(ctx, node):
        func = visit_stmt(ctx, node.func)
        args = visit_stmts(ctx, node.args)
        call = Call(func.name, args)
        if func.name == "new_closure":
            call.closure = True
        if node.func.id not in Record.log:
            if ctx.is_var_declared(func.name) and ctx.get_var_by_name(func.name) == PrimType.Closure:
                call.closure = True
            else:
                raise Exception(f"Can not find function {node.func.id} declared")
        return call


visit_stmt = ASTVisitor()


def visit_stmts(ctx, stmts):
    body = []
    for stmt in stmts:
        if ctx.returned != ReturnStatus.NoReturn or ctx.loop_status() != LoopStatus.Normal:
            break
        else:
            body.append(visit_stmt(ctx, stmt))
    return body