#include "ast/codegen.h"
#include <pythonrun.h>

namespace p2c {

    // void codegen_stmts(asdl_seq *body) {

    // }

    // void test_mod(mod_ty mod) {
    //     switch (mod->kind)
    //     {
    //     case Module_kind:
    //         codegen_stmts(mod->v.Module.body);
    //         break;
    //     case Expression_kind:
                
    //         break;
    //     default:
    //         break;
    //     }
    // }
    int test_code(std::string code, std::string filename) {
        // Py_Initialize();
        PyCompilerFlags cf = {0, 9};
        _Py_Identifier id = {NULL, filename.c_str(), NULL};
        PyObject* py_filename = _PyUnicode_FromId(&id);
        PyArena *arena = PyArena_New();
        mod_ty ast = PyParser_ASTFromStringObject(code.c_str(), py_filename, 257 /*Py_file_input*/, &cf, arena);
        // Py_Finalize();
        return 1;
    }
    
} // namespace p2c
