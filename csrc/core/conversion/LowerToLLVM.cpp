#include "core/conversion/LowerToLLVM.h"

#include "core/dialect/FirDialect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

namespace {

struct FIRToLLVMLoweringPass : 
    public mlir::PassWrapper<FIRToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::StandardOpsDialect, mlir::scf::SCFDialect, mlir::AffineDialect>();
    }
    void runOnOperation() final;
};

void FIRToLLVMLoweringPass::runOnOperation() {
    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

    mlir::LLVMTypeConverter typeConverter(&getContext());

    mlir::OwningRewritePatternList patterns;
    // mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
    // mlir::populateLoopToStdConversionPatterns(patterns, &getContext());
    mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

    auto module = getOperation();
    if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

} // namespace

namespace p2c {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
    return std::make_unique<FIRToLLVMLoweringPass>();
}
} // namespace p2c