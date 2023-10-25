import builtins
from typing import Any
from enum import Enum


class Builder:
    def __call__(self, ctx, node) -> Any:
        method = getattr(self, "visit_" + node.__class__.__name__, None)
        try:
            if method is None:
                error_msg = f'Unsupported node "{node.__class__.__name__}"'
                raise Exception(error_msg)
            # info 
            return method(ctx, node)
        except Exception as e:
            print(e)
            raise e
        
class VariableScopeGuard:
    def __init__(self, ctx) -> None:
        self.ctx = ctx
    
    def __enter__(self):
        self.ctx.local_scopes.append({})
    
    def __exit__(self, type, value, traceback):
        self.ctx.local_scopes.pop()


class ReturnStatus(Enum):
    NoReturn = 0
    ReturnedVoid = 1
    ReturnedValue = 2


class LoopStatus(Enum):
    Normal = 0
    Break = 1
    Continue = 2


class ASTContext:
    
    def __init__(self,
                 func=None,
                 global_vars=None,
                 argument_data=None,
                 file=None,
                 src=None,
                 start_line=None,
                ) -> None:
        self.func = func
        self.local_scopes = []
        self.loop_scopes = []
        self.returns = None
        self.global_vars = global_vars
        self.argument_data = argument_data
        self.return_data = None
        self.file = file
        self.src = src
        self.indent = 0
        for space in self.src[0]:
            if space == ' ':
                self.indent += 1
            else:
                break
        self.line_offset = start_line - 1
        self.raised = False
        self.returned = ReturnStatus.NoReturn
        self.closure = False
        self.return_ty = None

    def current_scope(self):
        return self.local_scopes[-1]

    def is_var_declared(self, name):
        for s in self.local_scopes:
            if name in s:
                return True
        return False
    
    def is_var_prevs(self, name):
        for s in self.local_scopes[0:-1]:
            if name in s:
                return True
        return False

    def create_variable(self, name, var):
        if name in self.current_scope():
            raise Exception(f'Redefinede variable {name}')
        self.current_scope()[name] = var

        
    def variable_scope_guard(self):
        return VariableScopeGuard(self)
    
    def loop_status(self):
        if self.loop_scopes:
            return self.loop_scopes[-1].status
        return LoopStatus.Normal
    
    def get_var_by_name(self, name) -> Any:
        for s in reversed(self.local_scopes):
            if name in s:
                return s[name]
        if name in self.global_vars:
            return self.global_vars[name]
        try:
            return getattr(builtins, name)
        except AttributeError:
            raise Exception(f'Variable "{name}" not found')
