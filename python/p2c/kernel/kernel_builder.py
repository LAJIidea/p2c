from p2c.ast.ast_utils import ASTContext
from p2c.ir.primitive_type import PrimType
from p2c.ir.record import Record
from p2c.ir.stmt import If, For, While, Assign, Decl, End, Ended, AugOp, Closure
from p2c.ir.expr import Identifier, Num, String, Boolean, Call, Binary, Unary, BinaryOp, UnaryOp, Array


def primtype_to_ctype(ty):
    if ty is PrimType.Int:
        return "int"
    if ty is PrimType.Float:
        return "float"
    if ty is PrimType.Array:
        return "array"
    if ty is PrimType.Bool:
        return "bool"
    if ty is PrimType.String:
        return "char *"
    if ty is PrimType.Function:
        return ""
    if ty is PrimType.Closure:
        return "closure*"
    if ty is None:
        return "void"


class KernelASTBuilder:
    def __init__(self, ctx: ASTContext) -> None:
        self.ctx = ctx
        self.ident = ''

    def codegen_module(self, mod):
        pass

    def codegen_func(self, func):
        name = func.name
        ret = func.returned
        parameters = func.parameters
        param_list = ''
        if name in Record.closure:
            param_list += "closure* cl, "
        for param in parameters:
            param_list += f"{primtype_to_ctype(param.ty)} {param.name}, "
        param_list = param_list[0:-2]
        print(f"{primtype_to_ctype(ret)} {name}({param_list}) " + "{")
        self.ident = '  '
        for stmt in func.stmts:
            self.codegen_stmt(stmt)
        print("}")

    def codegen_stmt(self, stmt):
        if isinstance(stmt, Decl):
            self.codegen_decl(stmt)
        elif isinstance(stmt, If):
            self.codegen_if(stmt)
        elif isinstance(stmt, For):
            self.codegen_for(stmt)
        elif isinstance(stmt, While):
            self.codegen_while(stmt)
        elif isinstance(stmt, Assign):
            self.codegen_assign(stmt)
        elif isinstance(stmt, Closure):
            self.codegen_closure(stmt)
        else:
            self.codegen_end(stmt)

    def codegen_decl(self, stmt):
        decl = primtype_to_ctype(stmt.ty) + " "
        for k, v in stmt.pairs.items():
            if isinstance(v, Array):
                decl += f'{k.name}[{len(v.elements)}] = ' + "{" + self.codegen_expr(v) + "}, "
            else:
                decl += f'{k.name} = {self.codegen_expr(v)}, '
        print(self.ident + decl[0:-2] + ";")

    def codegen_if(self, stmt):
        expr = self.codegen_expr(stmt.expr)
        print(self.ident + f"if ({expr}) " + "{")
        self.ident = self.ident + '  '
        for st in stmt.body:
            self.codegen_stmt(st)
        self.ident = self.ident[:-2]
        if len(stmt.elbody) != 0:
            print(self.ident + "} else {")
            self.ident = self.ident + '  '
            for elst in stmt.elbody:
                self.codegen_stmt(elst)
            self.ident = self.ident[:-2]
        print(self.ident + "}")

    def codegen_for(self, stmt):
        print(f"{self.ident}for (int lower = 0; lower < {stmt.size}; lower += {stmt.step}) " + "{")
        self.ident = self.ident + '  '
        print(f"{self.ident}{primtype_to_ctype(stmt.init.ty)} {stmt.init.name} = {stmt.bound}[lower];")
        for st in stmt.body:
            self.codegen_stmt(st)
        self.ident = self.ident[:-2]

    def codegen_while(self, stmt):
        expr = self.codegen_expr(stmt.expr)
        print(self.ident + f"while ({expr}) " + "{")
        self.ident = self.ident + '  '
        for st in stmt.body:
            self.codegen_stmt(st)
        self.ident = self.ident[:-2]
        print(self.ident + "}")

    def codegen_assign(self, stmt):
        expr = self.codegen_expr(stmt.right)
        target = self.codegen_expr(stmt.left)
        op = {
            AugOp.Add: "+=",
            AugOp.Sub: "-=",
            AugOp.Times: "*=",
            AugOp.Div: "/=",
            AugOp.Base: "=",
            AugOp.Bit: "%="
        }.get(stmt.op)
        print(self.ident + f"{target} {op} {expr};")

    def codegen_end(self, stmt):
        if stmt.ended is Ended.Continue:
            print(self.ident + "contine;")
        elif stmt.ended is Ended.Break:
            print(self.ident + "break")
        elif stmt.ended is Ended.Return:
            expr = self.codegen_expr(stmt.init)
            print(f"{self.ident}return {expr};")
        elif stmt.ended is Ended.Pass:
            pass

    def codegen_closure(self, stmt):
        fn = stmt.func.name
        data = stmt.data
        print(f"{self.ident}closure* new_closure = malloc(sizeof(closure));")
        print(f"{self.ident}new_closure->fun = {fn};")
        for k in data:
            print(f"{self.ident}new_closure->{k.name} = {k.name};")


    def codegen_expr(self, expr) -> str:
        if isinstance(expr, Identifier):
            return self.codegen_id(expr)
        if isinstance(expr, Num):
            return self.codegen_num(expr)
        if isinstance(expr, String):
            return self.codegen_str(expr)
        if isinstance(expr, Boolean):
            return self.codegen_bool(expr)
        if isinstance(expr, Call):
            return self.codegen_call(expr)
        if isinstance(expr, Binary):
            return self.codegen_binary(expr)
        if isinstance(expr, Unary):
            return self.codegen_unary(expr)
        if isinstance(expr, Array):
            return self.codegen_array(expr)
    
    def codegen_id(self, expr) -> str:
        if expr.closure:
            return f'new_closure->{expr.name}'
        return expr.name

    def codegen_num(self, expr) -> str:
        return str(expr.num)

    def codegen_str(self, expr) -> str:
        return f'\"{expr.num}\"'

    def codegen_bool(self, expr) -> str:
        if expr.val:
            return "true"
        else:
            return "false"
    
    def codegen_call(self, expr) -> str:
        args = []
        for arg in expr.args:
            args.append(self.codegen_expr(arg))
        arguments = ', '.join(args)
        if expr.closure:
            arguments = f'{expr.name}, ' + arguments if len(arguments) != 0 else f'{expr.name}'
            call = f'{expr.name}->fun({arguments})'
        call = f'{expr.name}({arguments})'
        return call

    def codegen_binary(self, expr) -> str:
        left = self.codegen_expr(expr.left)
        right = self.codegen_expr(expr.right)
        if expr.op is BinaryOp.Star:
            return f"pow({left}, {right})"
        if expr.op is BinaryOp.Floor:
            return f"floor({left} / {right})"
        op = {
            BinaryOp.Add: "+",
            BinaryOp.Sub: "-",
            BinaryOp.Times: "*",
            BinaryOp.Div: "/",
            BinaryOp.Mod: "%",
            BinaryOp.BitAnd: "&",
            BinaryOp.BitOr: "|",
            BinaryOp.BitXor: "^",
            BinaryOp.LShift: "<<",
            BinaryOp.RShift: ">>",
            BinaryOp.Equal: "==",
            BinaryOp.NotEqual: "!=",
            BinaryOp.GE: ">=",
            BinaryOp.LE: "<=",
            BinaryOp.Greater: ">",
            BinaryOp.Lesser: "<",
            BinaryOp.And: "&&",
            BinaryOp.Or: "||"
        }.get(expr.op)
        return f'{left} {op} {right}'

    def codegen_unary(self, expr) -> str:
        op = {
            UnaryOp.Bang: "!",
            UnaryOp.Minus: "-",
            UnaryOp.Tiled: "~"
        }.get(expr.op)
        e = self.codegen_expr(expr)
        return op + e
    
    def codegen_array(self, expr) -> str:
        elements = []
        for element in expr.elements:
            elements.append(self.codegen_expr(element))
        return ', '.join(elements)

    def dump(self, ir):
        print("#include<stdio.h>")
        print("#include <math.h>")
        for k, v in Record.closure.items():
            print("typedef struct closure {")
            parameters = v.func.parameters
            args = "struct closure* cl, "
            for param in parameters:
                args += f"{primtype_to_ctype(param.ty)} {param.name}, "
            print(f"  void (*fun)({args[0:-2]});")
            for var in v.data:
                print(f"  {primtype_to_ctype(var.ty)} {var.name};")
            print("} closure;")
            self.codegen_func(v.func)

        self.codegen_func(ir.body[0])