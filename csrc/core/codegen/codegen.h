#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>


#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace p2c {
#include <Python-ast.h>

mlir::ModuleOp codegen(mod_ty mod);

}