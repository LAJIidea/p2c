add_subdirectory(core)

file(GLOB BIND_SRC
    python/*.cpp
    )

pybind11_add_module(p2c_bind NO_EXTRAS ${BIND_SRC})

# These commands should apply to the DLL that is loaded from python, not the OBJECT library.
if (MSVC)
    set_property(TARGET p2c_bind APPEND PROPERTY LINK_FLAGS /DEBUG)
endif ()

target_link_libraries(p2c_bind PRIVATE p2c_core)