#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TPCIndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <memory>

namespace p2c {
class CpuJIT {
private:
    std::unique_ptr<llvm::orc::TargetProcessControl> TPC;
    std::unique_ptr<llvm::orc::ExecutionSession> ES;
    std::unique_ptr<llvm::orc::TPCIndirectionUtils> TPCIU;

    llvm::DataLayout DL;
    llvm::orc::MangleAndInterner Mangle;

    llvm::orc::RTDyldObjectLinkingLayer ObjectLayer;
    llvm::orc::IRCompileLayer CompileLayer;
    llvm::orc::IRTransformLayer OptimizeLayer;

    llvm::orc::JITDylib &MainJD;

    static void handleLazyCallThroughError() {
        llvm::errs() << "Failed to lazily compile function for call-through!\n";
        exit(1);
    }
public:
    CpuJIT(std::unique_ptr<llvm::orc::TargetProcessControl> TPC,
           std::unique_ptr<llvm::orc::ExecutionSession> ES,
           std::unique_ptr<llvm::orc::TPCIndirectionUtils> TPCIU,
           llvm::orc::JITTargetMachineBuilder JTMB, llvm::DataLayout DL)
        : TPC(std::move(TPC)), ES(std::move(ES)), TPCIU(std::move(TPCIU)),
          DL(std::move(DL)), Mangle(*this->ES, this->DL),
          ObjectLayer(*this->ES,
                      []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
          CompileLayer(*this->ES, ObjectLayer,
                       std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(JTMB))),
          OptimizeLayer(*this->ES, CompileLayer, optimizeModule),
          MainJD(this->ES->createBareJITDylib("<main>")) {
        MainJD.addGenerator(
            llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(DL.getGlobalPrefix())));
    }

    ~CpuJIT() {
        if (auto Err = ES->endSession())
            ES->reportError(std::move(Err));
        if (auto Err = TPCIU->cleanup())
            ES->reportError(std::move(Err));
    }

    static llvm::Expected<std::unique_ptr<CpuJIT>> Create() {
        auto SSP = std::make_shared<llvm::orc::SymbolStringPool>();
        auto TPC = llvm::orc::SelfTargetProcessControl::Create(SSP);
        if (!TPC)
            return TPC.takeError();
        
        auto ES = std::make_unique<llvm::orc::ExecutionSession>(std::move(SSP));

        auto TPCIU = llvm::orc::TPCIndirectionUtils::Create(**TPC);
        if (!TPCIU)
            return TPCIU.takeError();

        (*TPCIU)->createLazyCallThroughManager(
            *ES, llvm::pointerToJITTargetAddress(&handleLazyCallThroughError));
        
        if (auto Err = llvm::orc::setUpInProcessLCTMReentryViaTPCIU(**TPCIU))
            return std::move(Err);
        
        llvm::orc::JITTargetMachineBuilder JTMB((*TPC)->getTargetTriple());

        auto DL = JTMB.getDefaultDataLayoutForTarget();
        if (!DL)
            return DL.takeError();

        return std::make_unique<CpuJIT>(std::move(*TPC), std::move(ES),
                                        std::move(*TPCIU), std::move(JTMB),
                                        std::move(*DL));
    }

    const llvm::DataLayout &getDataLayout() const { return DL; }

    llvm::orc::JITDylib &getMainJITDylib() { return MainJD; }

    llvm::Error addModule(llvm::orc::ThreadSafeModule TSM, llvm::orc::ResourceTrackerSP RT = nullptr) {
        if (!RT)
            RT = MainJD.getDefaultResourceTracker();

        return OptimizeLayer.add(RT, std::move(TSM));
    }

    llvm::Expected<llvm::JITEvaluatedSymbol> lookup(llvm::StringRef Name) {
        return ES->lookup({&MainJD}, Mangle(Name.str()));
    }    

private:
    static llvm::Expected<llvm::orc::ThreadSafeModule>
    optimizeModule(llvm::orc::ThreadSafeModule TSM,
                   const llvm::orc::MaterializationResponsibility &R) {
        TSM.withModuleDo([](llvm::Module &M) {
            auto FPM = std::make_unique<llvm::legacy::FunctionPassManager>(&M);
        
            FPM->add(llvm::createInstructionCombiningPass());
            FPM->add(llvm::createReassociatePass());
            FPM->add(llvm::createGVNPass());
            FPM->add(llvm::createCFGSimplificationPass());
            FPM->doInitialization();

            for (auto &F : M) {
                FPM->run(F);
            }
        });

        return std::move(TSM);
    }
};

}