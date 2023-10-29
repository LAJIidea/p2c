#include "mlir/IR/BuiltinOps.h"
#include "core/jit/cpu_jit.h"
#include <vector>
#include <map>

namespace p2c {

    class JITModule {
    public:
        JITModule() {}
        virtual ~JITModule() {}
        intptr_t jitCompile(mlir::ModuleOp module, const std::string &name);
    private:
        std::map<std::string, intptr_t> funcMap;
        // std::shared_ptr<CpuJIT> TheJIT;
    };
    // int runJit(mlir::ModuleOp module);
}