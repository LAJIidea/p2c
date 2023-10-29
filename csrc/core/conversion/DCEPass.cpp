#include "core/conversion/DCEPass.h"
#include "core/dialect/FirDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


namespace {

struct StandardDCEPattern : public mlir::ConversionPattern {
    explicit StandardDCEPattern(mlir::MLIRContext *context)
        : ConversionPattern(mlir::AddIOp::getOperationName(), 1, context) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        if (mlir::isa<mlir::CallOp>(op))
            return mlir::success();
        if (op->use_empty()) {
            rewriter.eraseOp(op);
            for (auto operand : op->getOperands()) {
                // if (mlir::isa<mlir::CallOp>(operand.getDefiningOp()))
                //     continue;
                bool remove = true;
                for (auto user : operand.getUsers()) {
                    if (user != op) {
                        remove = false;
                        break;
                    }
                }
                if (remove) {
                    rewriter.eraseOp(operand.getDefiningOp());
                }
            }
        }
        return mlir::success();
    }
};

struct DCEPass : public mlir::PassWrapper<DCEPass, mlir::OperationPass<mlir::ModuleOp>> {
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::p2c::FIRDialect, mlir::StandardOpsDialect>();
    }
    void runOnOperation() final;
};

void DCEPass::runOnOperation() {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::ConversionTarget target(*context);

    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<StandardDCEPattern>(context);
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

} // namespace

namespace p2c {
std::unique_ptr<mlir::Pass> createDCEPass() {
    return std::make_unique<DCEPass>();
}    
} // namesapce p2c