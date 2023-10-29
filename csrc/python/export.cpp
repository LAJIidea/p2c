#include "python/export.h"
#include "core/p2c.h"

namespace p2c {

void export_module(py::module& m) {
    py::class_<Backend, std::shared_ptr<Backend>>(m, "Backend")
        .def(py::init<>())
        .def("emitIR", &Backend::emitIR, "")
        .def("jitCompile", &Backend::jitCompile, "")
        .def("setOptimization", &Backend::setOptimization, "")
        .def("lookup", &Backend::lookup, "")
        .def("invoke", &Backend::invoke, "");
}

PYBIND11_MODULE(p2c_bind, m) {
    m.doc() = "Python to C jit compiler";
    export_module(m);
}

} // namespace p2c