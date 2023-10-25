from p2c.ast.ast_utils import ASTContext
from p2c.ir.primitive_type import PrimType
from p2c.ir.stmt import If, For, While, Assign, Decl, End, Ended
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
        pass

    def codegen_assign(self, stmt):
        pass

    def codegen_end(self, stmt):
        if stmt.ended is Ended.Continue:
            print(self.ident + "contine;")
        elif stmt.ended is Ended.Break:
            print(self.ident + "break")
        elif stmt.ended is Ended.Return:
            expr = self.codegen_expr(stmt.init)
            print(f"{self.ident}return {expr};")

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
        self.codegen_func(ir.body[0])