#ifndef FIR_DIALECT
#define FIR_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// FIR dialect definition.
//===----------------------------------------------------------------------===//

def FIR_Dialect : Dialect {
  let name = "fir";
  let summary = "A MLIR dialect for python fast optimize ir";
  let description = [{
    This dialect is provides a fast optimize ir for python.
  }];
  let cppNamespace = "::mlir::p2c";
}

//===----------------------------------------------------------------------===//
// Base Operation defitions.
//===----------------------------------------------------------------------===//

class FIR_Op<string>

#endif // FIR_DIALECT