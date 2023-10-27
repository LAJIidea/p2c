#ifndef FIR_OPS
#define FIR_OPS

include "FirDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"

//===----------------------------------------------------------------------===//
// FIR Types.
//===----------------------------------------------------------------------===//

def ClosureType : Type<CPred<"$_self.isa<::mlir::p2c::ClosureType>()">,
                             "closure type", "::mlir::p2c::FunctionType">;

//===----------------------------------------------------------------------===//
// FIR Operations.
//===----------------------------------------------------------------------===//
def FIR_PassOp : FIR_Op<"pass"> {
    let summary = "pass operation";
    let description = [{
        Pass operation
    }];
    let results = (outs NoneType);
}

def FIR_YieldOp : FIR_Op<"yield"> {
    let summary = "Yiled operator";

    let description = [{
        The `fir.yield` operation is used to yield back
    }];
    let arguments = (ins AnyType:$value);
    let results = (outs AnyType:$res);
}

def FIR_WithOp : FIR_Op<"with"> {
    let summary = "With operator";

    let description = [{
        The `fir.with` operation is used to yield back
    }];
    let arguments = (ins AnyType:$value);
    let regions = (region SizedRegion<1>:$region);
}

def FIR_DecorateOp : FIR_Op<"decorate"> {
    let summary = "Decorate operator";

    let description = [{
        The `fir.decorate` operation is used to yield back
    }];
    let arguments = (ins FunctionType:$func);
    let results = (outs AnyType:$value);
}

def FIR_LambdaOp : FIR_Op<"lambda"> {
    let summary = "Lambda operator";

    let description = [{
        The `fir.lambda` is represent a lambda function
    }];
    let arguments = (Variadic<AnyType>:$args);
    let regions = (region SizedRegion<1>:$body);
    let results = (outs AnyType:$res);
}
#endif // FIR_OPS