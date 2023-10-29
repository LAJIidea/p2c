get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_library(p2c_jit runner.cpp)
target_link_libraries(p2c_jit
    PRIVATE
    p2c_codegen
    p2c_conversion
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTargetLLVMIR
    MLIRTransforms
    LLVMCore
    LLVMSupport
    LLVMOrcJIT
    LLVMCodeGen
    LLVMX86CodeGen
)