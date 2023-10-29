#include "core/dialect/FirDialect.h"
#include "core/dialect/FirOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/IR/Builders.h"

// using namespace mlir;
using namespace mlir::p2c;

// namesapce mlir {
// namespace p2c {
void FIRDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "core/dialect/FirOps.cpp.inc"
    >();
}
// } // namespace p2c
// } // namespace mlir