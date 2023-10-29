import inspect
import types
import typing
import textwrap
import ast
from typing import Any
from p2c.kernel.kernel_arguments import KernelArgument
from p2c.kernel.kernel_builder import KernelASTBuilder
from p2c.ast.ast_utils import ASTContext
from p2c.ast.transform import transform_ast
from p2c.ir.record import Record


# beautilful to handler closure, but not suitable this pretask
def get_global_vars(_func):
    global_vars = _func.__globals__.copy()

    freevar_names = _func.__code__.co_freevars
    closure = _func.__closure__
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value
    return global_vars


def get_tree_and_ctx(
    kernel,
    arg_features=None,
    args=None,
):
    file = inspect.getsourcefile(kernel.func)
    src, start_line = inspect.getsourcelines(kernel.func)
    src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
    tree = ast.parse(textwrap.dedent("\n".join(src)))

    func_body = tree.body[0]
    func_body.decorator_list = []

    global_vars = get_global_vars(kernel.func)

    return tree, ASTContext(
        func=kernel,
        global_vars=global_vars,
        argument_data=args,
        file=file,
        src=src,
        start_line=start_line
    )


class translate:
    counter = 0

    def __init__(self, _func, classkernel=None, enable_jit=False) -> None:
        self.func = _func
        self.kernel_counter = translate.counter
        translate.counter += 1
        self.arguments = []
        self.return_type = None
        # self.classkernel = _classkernel
        self.extract_arguments()
        self.compile()
        Record.kernels.append(self)
        self.banckend = None

    def enable_jit(self):
        from .p2c_bind import Backend
        self.backend = Backend()

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect._empty, None):
            self.return_type = sig.return_annotation
            # if isinstance(self.return_type, (types.GenericAlias, typing._GebericAlias)) and self.return_type.__origin__ is tuple:
            #     self.return_type = self.return_type.__args__
            # if not isinstance(self.return_type, (list, tuple)):
            #     self.return_type = (self.return_type)
            # for return_type in self.return_type:
            #     if return_type is Ellipsis:
            #         pass
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                pass
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                pass
            if param.default is not inspect.Parameter.empty:
                pass
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                pass
            annotation = param.annotation
            if param.annotation is inspect.Parameter.empty:
                pass
            else:
                self.arguments.append(KernelArgument(annotation, param.name, param.default))

    def compile(self) -> None:
        tree, ctx = get_tree_and_ctx(self, arg_features=None, args=None)
        self.ir = transform_ast(tree, ctx)
    
    def dump(self) -> None:
        builder = KernelASTBuilder()
        builder.dump(self.ir)

    def dumpMLIR(self) -> None:
        if self.backend is not None:
            src = inspect.getsource(self.func)
            file = inspect.getsourcefile(self.func)
            self.backend.emitIR(src, file)
        else:
            print("Please enable jit to dump MLIR")
    
    def construct_arg_dict(self, *args):
        args_dict = {}
        for i, arg_name in enumerate(self.arguments):
            args_dict[arg_name.name] = args[i]
        return args_dict

    def jit(self, args) -> None:
        if self.backend is not None:
            # args_dict = self.construct_arg_dict(*args)
            if self.backend.lookup(self.func.__name__):
                return self.backend.invoke(self.func.__name__, args)
            else:
                src = inspect.getsource(self.func)
                file = inspect.getsourcefile(self.func)
                self.backend.jitCompile(src, file, self.func.__name__)
                return self.backend.invoke(self.func.__name__, args)
        else:
            print("Please enable jit to compile")

    def optimizeDCE(self) -> None:
        if self.backend is not None:
            self.backend.setOptimization(True)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
