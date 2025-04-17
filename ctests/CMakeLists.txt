add_executable(test_triton_add test_triton_add.cpp)
target_link_libraries(test_triton_add PRIVATE Torch::Torch operators)
# executable are installed to "${CMAKE_INSTALL_PREFIX}/bin",
# so they links libraries in "../lib" for portability
set_target_properties(test_triton_add PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/../lib"
  BUILD_WITH_INSTALL_RPATH FALSE
)

add_executable(test_triton_sum test_triton_sum.cpp)
target_link_libraries(test_triton_sum PRIVATE Torch::Torch operators)
set_target_properties(test_triton_sum PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/../lib"
  BUILD_WITH_INSTALL_RPATH FALSE
)

install(TARGETS test_triton_add test_triton_sum)
