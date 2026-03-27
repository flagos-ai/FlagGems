# MUSA backend: force-link libtorch_python.so (needed by libmusa_kernels.so
# at runtime) and allow unresolved MPI/numa symbols from libtorch_cpu.so.
if(FLAGGEMS_BACKEND STREQUAL "MUSA")
    link_libraries("-Wl,--no-as-needed" ${torch_python_lib} "-Wl,--as-needed" ${Python_LIBRARIES})
    add_link_options("-Wl,--allow-shlib-undefined")
endif()

if(FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP)
  add_executable(test_triton_add test_triton_add.cpp)
  target_link_libraries(test_triton_add
    PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_triton_add PRIVATE FLAGGEMS_POINTWISE_DYNAMIC)
  add_test(NAME test_triton_add COMMAND test_triton_add)

  add_executable(test_triton_div test_triton_div.cpp)
  target_link_libraries(test_triton_div
    PRIVATE Torch::Torch operators gtest gtest_main)
  target_compile_definitions(test_triton_div PRIVATE FLAGGEMS_POINTWISE_DYNAMIC)
  add_test(NAME test_triton_div COMMAND test_triton_div)

  add_executable(test_triton_fill test_triton_fill.cpp)
  target_link_libraries(test_triton_fill
    PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
  target_compile_definitions(test_triton_fill PRIVATE FLAGGEMS_POINTWISE_DYNAMIC)
  add_test(NAME test_triton_fill COMMAND test_triton_fill)
endif()

add_executable(test_triton_reduction test_triton_reduction.cpp)
target_link_libraries(test_triton_reduction
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_reduction COMMAND test_triton_reduction)

add_executable(test_triton_mm test_triton_mm.cpp)
target_link_libraries(test_triton_mm
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_mm COMMAND test_triton_mm)

add_executable(test_triton_addmm test_triton_addmm.cpp)
target_link_libraries(test_triton_addmm
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_addmm COMMAND test_triton_addmm)

add_executable(test_triton_bmm test_triton_bmm.cpp)
target_link_libraries(test_triton_bmm
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_bmm COMMAND test_triton_bmm)

add_executable(test_triton_norm test_triton_norm.cpp)
target_link_libraries(test_triton_norm
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_norm COMMAND test_triton_norm)

add_executable(test_triton_rope test_triton_rope.cpp)
target_link_libraries(test_triton_rope
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_rope COMMAND test_triton_rope)

add_executable(test_triton_topk test_triton_topk.cpp)
target_link_libraries(test_triton_topk
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_topk COMMAND test_triton_topk)

add_executable(test_triton_contiguous test_triton_contiguous.cpp)
target_link_libraries(test_triton_contiguous
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_contiguous COMMAND test_triton_contiguous)

add_executable(test_triton_cat test_triton_cat.cpp)
target_link_libraries(test_triton_cat
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_cat COMMAND test_triton_cat)

add_executable(test_triton_zeros_constructor test_triton_tensor_constructor.cpp)
target_link_libraries(test_triton_zeros_constructor
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_zeros_constructor COMMAND test_triton_zeros_constructor)

add_executable(test_triton_embedding test_triton_embedding.cpp)
target_link_libraries(test_triton_embedding
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_embedding COMMAND test_triton_embedding)

add_executable(test_triton_argmax test_triton_argmax.cpp)
target_link_libraries(test_triton_argmax
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_argmax COMMAND test_triton_argmax)

add_executable(test_triton_sort test_triton_sort.cpp)
target_link_libraries(test_triton_sort
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_sort COMMAND test_triton_sort)

add_executable(test_triton_exponential_ test_triton_exponential_.cpp)
target_link_libraries(test_triton_exponential_
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_exponential_ COMMAND test_triton_exponential_)

add_executable(test_triton_softmax test_triton_softmax.cpp)
target_link_libraries(test_triton_softmax
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main ${Python_LIBRARIES} pybind11::embed)
add_test(NAME test_triton_softmax COMMAND test_triton_softmax)

add_executable(test_triton_reshape_and_cache_flash test_triton_reshape_and_cache_flash.cpp)
target_link_libraries(test_triton_reshape_and_cache_flash
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_reshape_and_cache_flash COMMAND test_triton_reshape_and_cache_flash)

add_executable(test_triton_flash_attn_varlen test_triton_flash_attn_varlen.cpp)
target_link_libraries(test_triton_flash_attn_varlen
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_flash_attn_varlen COMMAND test_triton_flash_attn_varlen)

add_executable(test_triton_rwkv_ka_fusion test_triton_rwkv_ka_fusion.cpp)
target_link_libraries(test_triton_rwkv_ka_fusion
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_rwkv_ka_fusion COMMAND test_triton_rwkv_ka_fusion)

add_executable(test_triton_rwkv_mm_sparsity test_triton_rwkv_mm_sparsity.cpp)
target_link_libraries(test_triton_rwkv_mm_sparsity
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_rwkv_mm_sparsity COMMAND test_triton_rwkv_mm_sparsity)

add_executable(test_triton_copy test_triton_copy.cpp)
target_link_libraries(test_triton_copy
  PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main)
add_test(NAME test_triton_copy COMMAND test_triton_copy)

# Set FLAGGEMS_SOURCE_DIR for all tests that load Triton kernels at runtime.
# Compute it from the project source directory.
set(_FLAGGEMS_SRC_DIR "${CMAKE_SOURCE_DIR}/src/flag_gems")
get_property(_all_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
foreach(_t IN LISTS _all_tests)
  set_tests_properties(${_t} PROPERTIES
    ENVIRONMENT "FLAGGEMS_SOURCE_DIR=${_FLAGGEMS_SRC_DIR}")
endforeach()
