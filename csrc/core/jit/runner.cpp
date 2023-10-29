#include "core/jit/runner.h"
#include "core/jit/cpu_jit.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/StringRef.h"
#include <iostream>

namespace p2c {

static std::unique_ptr<llvm::LLVMContext> llvmContext;

// JITModule::JITModule() : TheJIT(CpuJIT::Create().get().get()) {}

intptr_t JITModule::jitCompile(mlir::ModuleOp module, const std::string &name) {
    // llvm::LLVMContext llvmContext;
    // auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

    // if (!llvmModule) {
    //     llvm::errs() << "Failed to emit LLVM IR\n";
    //     return -1;
    // }
    // // llvmModule->setDataLayout(TheJIT->getDataLayout());
    // llvm::InitializeNativeTarget();
    // llvm::InitializeNativeTargetAsmPrinter();
    // mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());
    // auto RT = TheJIT->getMainJITDylib().createResourceTracker();

    // auto TSM = llvm::orc::ThreadSafeModule(std::move(llvmModule), 
    //         std::move(std::make_unique<llvm::LLVMContext>()));
    // auto err = TheJIT->addModule(std::move(TSM), RT);
    // auto symOrErr = TheJIT->lookup(name);
    // if (!symOrErr) {
    //     llvm::errs() << "failed to load function\n";
    //     return -1;
    // }
    // auto sym = symOrErr.get();
    // intptr_t func = (intptr_t)sym.getAddress();
    // return func;

    return 0;
}

// bool JITModule::lookup(const std::string &funcName) {
//     return funcMap.count(funcName);
// }

// int JITModule::invoke(const std::string &funcName, void **args) {
//     if (!lookup(funcName)) {
//         llvm::errs() << "function not found\n";
//         return -1;
//     }
//     auto *func = (int (*)(void **))(funcMap[funcName]);
//     return func(args);
// }

int runJit(mlir::ModuleOp module) {
    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    auto maybeEngine = mlir::ExecutionEngine::create(
        module, /*llvmModuleBuilder=*/nullptr, optPipeline);
    auto &engine = maybeEngine.get();  

    // Invoke the JIT-compiled function.
    int arg1 = 5;
    void *args[] = {&arg1};
    // llvm::SmallVector<void *> args(1, &arg1);
    // auto invocationResult = engine->invoke("test", args); 
    // std::cout << *(&arg1) << std::endl;
    auto maybeFunc = engine->lookup("main");
    auto func = maybeFunc.get();
    if (func) {
        std::cout << func << std::endl;
        func(args);
    }
    // int arg1 = 5;
    // void *args[] = {&arg1};
    // func(args);
    // auto baseFunc = reinterpret_cast<int (*)(void **)>(func);
    // int res = baseFunc(args);
    return 0;
}

} // namespace p2c