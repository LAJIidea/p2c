from typing import Any


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


class ASTContext:
    
    def __init__(self,
                 func=None,
                 global_vars=None,
                 argument_data=None,
                 file=None,
                 src=None,
                 start_line=None
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
