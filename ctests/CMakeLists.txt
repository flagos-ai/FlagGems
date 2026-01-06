add_executable(test_triton_pointwise test_triton_pointwise.cpp)
target_link_libraries(test_triton_pointwise
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_pointwise COMMAND test_triton_pointwise)

add_executable(test_triton_reduction test_triton_reduction.cpp)
target_link_libraries(test_triton_reduction
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_reduction COMMAND test_triton_reduction)

add_executable(test_triton_blas test_triton_blas.cpp)
target_link_libraries(test_triton_blas PRIVATE Torch::Torch operators gtest gtest_main)
add_test(NAME test_triton_blas COMMAND test_triton_blas)

add_executable(test_triton_norm test_triton_norm.cpp)
target_link_libraries(test_triton_norm
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_norm COMMAND test_triton_norm)
