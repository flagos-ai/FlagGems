add_executable(test_triton_pointwise test_triton_pointwise.cpp)
target_link_libraries(test_triton_pointwise
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
# executable are installed to "${CMAKE_INSTALL_PREFIX}/bin",
# so they links libraries in "../lib" for portability
set_target_properties(test_triton_pointwise PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/../lib"
  BUILD_WITH_INSTALL_RPATH FALSE
)
add_test(NAME test_triton_pointwise COMMAND test_triton_pointwise)

add_executable(test_triton_reduction test_triton_reduction.cpp)
target_link_libraries(test_triton_reduction
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
set_target_properties(test_triton_reduction PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/../lib"
  BUILD_WITH_INSTALL_RPATH FALSE
)
add_test(NAME test_triton_reduction COMMAND test_triton_reduction)

add_executable(test_triton_blas test_triton_blas.cpp)
target_link_libraries(test_triton_blas PRIVATE Torch::Torch operators gtest gtest_main)
set_target_properties(test_triton_blas PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/../lib"
  BUILD_WITH_INSTALL_RPATH FALSE
)
add_test(NAME test_triton_blas COMMAND test_triton_blas)
