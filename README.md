# p2c

这是一个将Python翻译成C的项目，项目结构为:
├─.vscode
├─csrc
│  ├─core
│  │  ├─codegen
│  │  ├─conversion
│  │  ├─dialect
│  │  ├─jit
│  │  ├─runtime
│  │  └─test
│  └─python
└─python
    └─p2c
        ├─ast 
        ├─ir
        └─kernel

其中python文件夹是项目的核心部分，也是可以独立执行部分，无需编译C项目即可单独运行。
python文件夹内为p2c文件夹和测试用例组成，该项目的核心功能为在运行时将python代码转译成C代码
使用方式为：导入p2c模块内的translate类，将其作为装饰器修饰在函数上，被修饰的Python函数会在
初始化时转译成等价的C语言函数。python根文件夹下的以_test为后缀的文件均为测试用例，可以直接在python目录下执行。
python独立部分功能均为使用特殊的第三方包，所复用的均为python内置库，如ast, typing等。
若想开启jit运行，则需要编译csrc。

csrc文件夹为C++项目，是p2c项目的jit编译后端，该项目是通过将Python代码Lower到MLIR来完成
一系列的优化，并最终通过LLVM JIT来执行。需要注意该项目必须使用Python3.9.8版本，通过pybind在C端复用
CPython的ast模块来Codegen MLIR。同时Win电脑中安装了python3.9.8，可以不用编译该项目。
pyd文件已经编译添加到了python模块中。
若想源码编译调试，编译命令在项目根路径的build.txt中。
该项目的依赖的第三方库为LLVM/MLIR(12.0.0), Pybind11, CPython(3.9.8).
csrc文件夹下的test文件夹内的py文件为测试用例，执行时，可将其移动到项目根路径下的python目录中执行。

.vscode文件提供了vscode cmake插件的编译命令和调试配置

参考资料：[nuitka](), [numba]()