#include "core/p2c.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "core/dialect/FirDialect.h"
#include "core/conversion/LowerToLLVM.h"
#include "core/conversion/DCEPass.h"
#include "core/jit/runner.h"
#include "core/codegen/codegen.h"

#include <iostream>


namespace p2c {

mod_ty parser(const std::string &code, const std::string filename) {
    PyCompilerFlags cf = {0, 9};
    _Py_Identifier id = {NULL, filename.c_str(), NULL};
    PyObject* py_filename = _PyUnicode_FromId(&id);
    PyArena* arena = PyArena_New();
    return PyParser_ASTFromStringObject(code.c_str(), py_filename, 257, &cf, arena);
}

void Backend::emitIR(const std::string &code, const std::string filename) {

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::p2c::FIRDialect>();
    context.loadDialect<mlir::StandardOpsDialect>();
    CodeGen codegen(context, filename);
    mlir::OwningModuleRef module = codegen.codegen(parser(code, filename));
    if (optimization) {
        mlir::PassManager pm(&context);
        mlir::applyPassManagerCLOptions(pm);
        pm.addPass(p2c::createDCEPass());
        if (mlir::failed(pm.run(*module)))
            return ;
    }
    module->dump();
}

void Backend::jitCompile(const std::string &code, const std::string &filename, const std::string &funcname) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::ExitOnError ExitOnErr;
    auto TheJIT = ExitOnErr( CpuJIT::Create());
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::p2c::FIRDialect>();
    context.loadDialect<mlir::StandardOpsDialect>();
    CodeGen codegen(context, filename);
    mlir::OwningModuleRef module = codegen.codegen(parser(code, filename));
    mlir::PassManager pm(&context);
    
    mlir::applyPassManagerCLOptions(pm);

    pm.addPass(p2c::createLowerToLLVMPass());

    if (mlir::failed(pm.run(*module)))
        return ;
    // JITModule jit;
    // intptr_t func = jit.jitCompile(*module, funcname);
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return ;
    }
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());
    auto RT = TheJIT->getMainJITDylib().createResourceTracker();

    auto TSM = llvm::orc::ThreadSafeModule(std::move(llvmModule), 
            std::move(std::make_unique<llvm::LLVMContext>()));
    ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
    auto sym = ExitOnErr(TheJIT->lookup(funcname));
    auto address = (intptr_t)sym.getAddress();
    std::function<int(int)> func = (int (*)(int))(address);
    funcMap.insert({funcname, func});
}

bool Backend::lookup(const std::string &funcName) {
    return funcMap.count(funcName);
}

int Backend::invoke(const std::string &funcName, int arg) {
    if (!lookup(funcName)) {
        std::cout << "Function " << funcName << " not found\n";
        return -1;
    }
    auto func = funcMap[funcName];
    return func(arg);
}
} // namespace p2c