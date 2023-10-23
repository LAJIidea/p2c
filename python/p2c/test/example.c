#include <Python.h>
#include <stdio.h>

PyObject* inner(PyObject* self, PyObject* args) {
    double m, n;
    int c;
    if (!PyArg_ParseTuple(args, "ddi", &m, &n, &c)) {
        return NULL;
    }
    return PyFloat_FromDouble(m * n + c);
}

PyObject* outer(PyObject* self, PyObject* args) {
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }
    int c = a * b + b;
    PyObject* inner_args = Py_BuildValue("dii", 1.2, 2.3, c);
    PyObject* result = inner(NULL, inner_args);
    Py_DECREF(inner_args);
    return result;
}

PyMethodDef module_methods[] = {
    {"outer", outer, METH_VARARGS, "Outer function"},
    {"inner", inner, METH_VARARGS, "Inner function"},
    {NULL, NULL, 0, NULL}
};

struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "module",
    "Module docstring",
    -1,
    module_methods
};

PyObject* PyInit_module(void) {
    return PyModule_Create(&module_def);
}

int main(int argc, char* argv[]) {
    PyImport_AppendInittab("module", &PyInit_module);
    Py_Initialize();
    PyImport_ImportModule("module");
    Py_Finalize();
    return 0;
}