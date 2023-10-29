#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <Python-ast.h>
#include <memory>
#include <string>
#include <unordered_map>
namespace p2c {

class CodeGen {
public:
    CodeGen(mlir::MLIRContext &context, const std::string &filename);
    mlir::OwningModuleRef codegen(mod_ty mod);
private:
    mlir::FuncOp codegen_func(stmt_ty func);
    mlir::LogicalResult codegen_stmts(asdl_seq* body);
    mlir::Value codegen_stmt(stmt_ty stmt);
    mlir::Value codegen_expr(expr_ty expr);
    llvm::SmallVector<mlir::Type> codegen_args(arguments_ty args);
    mlir::Value codegen_return(stmt_ty stmt);
    mlir::Value codegen_assign(stmt_ty stmt);
    mlir::Value codegen_for(stmt_ty stmt);
    mlir::Value codegen_while(stmt_ty stmt);
    mlir::Value codegen_if(stmt_ty stmt);
    mlir::Value codegen_expr_stmt(stmt_ty stmt);
    mlir::Value codegen_id(expr_ty expr);
    mlir::Value codegen_num(expr_ty expr);
    mlir::Value codegen_call(expr_ty expr);
    mlir::Value codegen_binop(expr_ty expr);
    mlir::Value codegen_unaryop(expr_ty expr);
    mlir::Type getType(expr_ty expr);
    mlir::Location loc(int line, int col);
    mlir::LogicalResult declare(std::string name, mlir::Value value);
    mlir::ModuleOp theModule;
    mlir::OpBuilder builder;
    std::string filename;
    // llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
    std::unordered_map<std::string, mlir::Value> symbolTable;
};

}