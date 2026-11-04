add_library(flag_gems_cpp SHARED
    flag_gems_ops.cpp
    binding.cpp
)
target_link_libraries(flag_gems_cpp PRIVATE ${ILUVATAR_RUNTIME_LIB})
target_include_directories(flag_gems_cpp PRIVATE ${FLAGGEMS_ROOT}/include)

# pybind11
find_package(pybind11 REQUIRED)
pybind11_add_module(flag_gems_binding binding.cpp)
target_link_libraries(flag_gems_binding PRIVATE flag_gems_cpp ${ILUVATAR_RUNTIME_LIB})
