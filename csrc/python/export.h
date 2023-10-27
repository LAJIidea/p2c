#pragma once

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/operators.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace p2c {
namespace py = pybind11;

void export_module(py::module& m);
} // namespace p2c