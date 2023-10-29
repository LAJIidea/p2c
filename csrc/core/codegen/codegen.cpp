#include "core/codegen/codegen.h"
#include <iostream>

namespace p2c {
CodeGen::CodeGen(mlir::MLIRContext &context, const std::string &filename) 
    : builder(&context), filename(filename) {}

mlir::Location CodeGen::loc(int line, int col) {
    return builder.getFileLineColLoc(builder.getIdentifier(filename), line, col);
}

mlir::OwningModuleRef CodeGen::codegen(mod_ty mod) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    switch (mod->kind)
    {
    case Module_kind: {
        Py_ssize_t i;
        asdl_seq *seq = mod->v.Module.body;
        for (i = 0; i < asdl_seq_LEN(seq); i++) {
            stmt_ty stmt = (stmt_ty)asdl_seq_GET(seq, i);
            auto func = codegen_func(stmt);
            theModule.push_back(func);
        }
        break;
    }
    case Interactive_kind:
        std::cout << "Interactive" << std::endl;
        break;
    case Expression_kind:
        std::cout << "Expression" << std::endl;
        break;
    default:
        break;
    }
    return theModule;
}

mlir::FuncOp CodeGen::codegen_func(stmt_ty func) {
    auto loc = this->loc(func->lineno, func->col_offset);
    if (func->kind != FunctionDef_kind) {
        std::cout << "Not a function" << std::endl;
        return nullptr;
    }
    auto types = codegen_args(func->v.FunctionDef.args);
    mlir::TypeRange ret;
    if (func->v.FunctionDef.returns) {
        ret = llvm::SmallVector<mlir::Type, 4>(1, getType(func->v.FunctionDef.returns));
    } else {
        ret = llvm::None;
    }
    auto func_type = builder.getFunctionType(types, ret);
    std::string funcName(PyUnicode_AsUTF8(func->v.FunctionDef.name));
    auto funcOp =  mlir::FuncOp::create(loc, funcName, func_type);
    auto& entrBlock = *funcOp.addEntryBlock();

    // Declare all the function arguments in the symbol table.
    for (size_t i = 0; i < entrBlock.getArguments().size(); i++) {
        auto arg = entrBlock.getArgument(i);
        arg_ty param = (arg_ty) asdl_seq_GET(func->v.FunctionDef.args->args, i);
        std::string argName(PyUnicode_AsUTF8(param->arg));
        if (mlir::failed(declare(argName, arg))) {
            return nullptr;
        }
    }

    builder.setInsertionPointToStart(&entrBlock);

    // Emit the body of the function.
    if (mlir::failed(codegen_stmts(func->v.FunctionDef.body))) {
        funcOp.erase();
        return nullptr;
    }
    return funcOp;
}

mlir::Type CodeGen::getType(expr_ty expr) {
    if (expr->kind != Name_kind) {
        std::cout << "Unknow type" << std::endl;
        return nullptr;
    }
    auto name = expr->v.Name.id;
    if (_PyUnicode_EqualToASCIIString(name, "int")) {
        return builder.getIntegerType(32);
    } else if (_PyUnicode_EqualToASCIIString(name, "float")) {
        return builder.getF32Type();
    } else if (_PyUnicode_EqualToASCIIString(name, "bool")) {
        return builder.getI1Type();
    } else {
        std::cout << "Unknow type" << std::endl;
        return nullptr;
    }
}

mlir::LogicalResult CodeGen::declare(std::string name, mlir::Value value) {
    if (symbolTable.count(name)) {
        std::cout << "Redeclaration of variable" << std::endl;
        return mlir::failure();
    }
    symbolTable.insert({name, value});
    return mlir::success();
}

llvm::SmallVector<mlir::Type> CodeGen::codegen_args(arguments_ty args) {
    llvm::SmallVector<mlir::Type, 4> argTypes;
    Py_ssize_t i;
    asdl_seq *seq = args->args;
    for (i = 0; i < asdl_seq_LEN(seq); i++) {
        arg_ty arg = (arg_ty)asdl_seq_GET(seq, i);
        if (!arg->annotation) {
            std::cout << "Unknow func type" << std::endl;
            return argTypes;
        }
        if (auto argType = getType(arg->annotation)) {
            argTypes.push_back(argType);
        } else {
            std::cout << "Unknow func type" << std::endl;
            return argTypes;
        }
    }
    return argTypes;
}

mlir::LogicalResult CodeGen::codegen_stmts(asdl_seq* body) {
    Py_ssize_t i;
    for (i = 0; i < asdl_seq_LEN(body); i++) {
        stmt_ty stmt = (stmt_ty)asdl_seq_GET(body, i);
        if (!codegen_stmt(stmt)) {
            return mlir::failure();
        }
    }
    return mlir::success();
}

mlir::Value CodeGen::codegen_stmt(stmt_ty statement) {
    switch (statement->kind)
    {
    case Return_kind:
        return codegen_return(statement);
    case Assign_kind:
        return codegen_assign(statement);
    case For_kind:
        return codegen_for(statement);
    case While_kind:
        return codegen_while(statement);
    case If_kind:
        return codegen_if(statement);
    case Expr_kind:
        return codegen_expr_stmt(statement);
    default:
        break;
    }
    return nullptr;
}

mlir::Value CodeGen::codegen_return(stmt_ty stmt) {
    auto loc = this->loc(stmt->lineno, stmt->col_offset);
    auto expr = stmt->v.Return.value;
    auto value = codegen_expr(expr);
    builder.create<mlir::ReturnOp>(loc, value);
    return value;
}

mlir::Value CodeGen::codegen_assign(stmt_ty stmt) {
    auto loc = this->loc(stmt->lineno, stmt->col_offset);
    auto value = codegen_expr(stmt->v.Assign.value);
    asdl_seq* seq = stmt->v.Assign.targets;
    expr_ty target = (expr_ty) asdl_seq_GET(seq, 0);
    std::string name(PyUnicode_AsUTF8(target->v.Name.id));
    // if (symbolTable.count(name)) {
    //     symbolTable[name] = value;
    // }
    symbolTable[name] = value;
    return value;
}

mlir::Value CodeGen::codegen_for(stmt_ty stmt) {
    return nullptr;
}

mlir::Value CodeGen::codegen_while(stmt_ty stmt) {
    return nullptr;
}

mlir::Value CodeGen::codegen_if(stmt_ty stmt) {
    return nullptr;
}

mlir::Value CodeGen::codegen_expr_stmt(stmt_ty stmt) {
    return nullptr;
}

mlir::Value CodeGen::codegen_expr(expr_ty expr) {
    switch (expr->kind) {
    case Name_kind:
        return codegen_id(expr);
    case Subscript_kind:
        std::cout << "Subcript" << std::endl;
        break;
    case Starred_kind:
        std::cout << "Starred" << std::endl;
        break;
    case List_kind:
        std::cout << "List" << std::endl;
        break;
    case BoolOp_kind:
        std::cout << "BoolOp" << std::endl;
        break;
    case BinOp_kind:
        return codegen_binop(expr);
    case UnaryOp_kind:
        return codegen_unaryop(expr);
    case Call_kind:
        return codegen_call(expr);
    case Compare_kind:
        std::cout << "Compare" << std::endl;
        break;
    case Constant_kind:
        return codegen_num(expr);
    default:
        break;
    }
    return nullptr;
}

mlir::Value CodeGen::codegen_id(expr_ty expr) {
    // if (auto variable = symbolTable.lookup(PyUnicode_AsUTF8(expr->v.Name.id))) {
    //     return variable;
    // }
    if (symbolTable.count(PyUnicode_AsUTF8(expr->v.Name.id))) {
        return symbolTable[PyUnicode_AsUTF8(expr->v.Name.id)];
    }
    return nullptr;
}

mlir::Value CodeGen::codegen_num(expr_ty expr) {
    auto loc = this->loc(expr->lineno, expr->col_offset);
    if (PyLong_Check(expr->v.Constant.value)) {
        auto ty = builder.getIntegerType(32);
        int value = PyLong_AsLong(expr->v.Constant.value);
        return builder.create<mlir::ConstantOp>(loc, ty, builder.getI32IntegerAttr(value));
    } else if (PyFloat_Check(expr->v.Constant.value)) {
        auto ty = builder.getF32Type();
        float value = PyFloat_AsDouble(expr->v.Constant.value);
        return builder.create<mlir::ConstantOp>(loc, ty, builder.getF32FloatAttr(value));
    }
    return nullptr;
}

mlir::Value CodeGen::codegen_call(expr_ty expr) {
    return nullptr;
}

mlir::Value CodeGen::codegen_binop(expr_ty expr) {
    auto loc = this->loc(expr->lineno, expr->col_offset);
    auto left = codegen_expr(expr->v.BinOp.left);
    auto right = codegen_expr(expr->v.BinOp.right);
    switch (expr->v.BinOp.op)
    {
        case Add:
            return builder.create<mlir::AddIOp>(loc, left, right);
        default:
            break;
    }
    return nullptr;
}

mlir::Value CodeGen::codegen_unaryop(expr_ty expr) {
    return nullptr;
}

} // namespace p2c