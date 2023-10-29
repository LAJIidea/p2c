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

add_llvm_library(p2c_codegen codegen.cpp)

target_link_libraries(p2c_codegen PUBLIC ${PYTHON_LIBRARIES}
    PRIVATE ${LIBS}
    MLIRAnalysis 
    MLIRCallInterfaces 
    MLIRIR
    MLIRParser 
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRLoopLikeInterface
    MLIRTransforms
)