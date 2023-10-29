#pragma once

#include <memory>
namespace mlir {
class Pass;
}

namespace p2c {

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace p2c