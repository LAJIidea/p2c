set(LLVM_TARGET_DEFINITIONS FirOps.td)
mlir_tablegen(FirOps.h.inc -gen-op-decls)
mlir_tablegen(FirOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS FirDialect.td)
mlir_tablegen(FirDialect.h.inc -gen-dialect-decls)

add_public_tablegen_target(MLIRFirOpsIncGen)

add_mlir_dialect_library(MLIRFir
    FirDialect.cpp
    FirOps.cpp

    DEPENDS
    MLIRFirOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)