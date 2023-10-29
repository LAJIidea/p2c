file(GLOB SRC *.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRFir
    MLIRTargetLLVMIR
    MLIRExecutionEngine
    LLVMLinker
)

add_mlir_library(p2c_conversion ${SRC}
    LINK_LIBS PUBLIC
    ${LIBS}
)