#ifndef FIR_DIALECT
#define FIR_DIALECT

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

class FIR_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<FIR_Dialect, mnemonic, traits>;

#endif // FIR_DIALECT