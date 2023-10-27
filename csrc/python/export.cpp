#include "python/export.h"
#include "ast/codegen.h"

namespace p2c {

void export_module(py::module& m) {
m.def("test_code", &test_code);
}

PYBIND11_MODULE(p2c_bind, m) {
    m.doc() = "Python to C jit compiler";
    export_module(m);
}

} // namespace p2c