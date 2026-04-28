add_library(operators
            SHARED
            device_info.cpp
            zeros.cpp
            utils.cpp
            sum.cpp
            max.cpp
            mm.cpp
            rms_norm.cpp
            fused_add_rms_norm.cpp
            rotary_embedding.cpp
            topk.cpp
            addmm.cpp
            nonzero.cpp
            contiguous.cpp
            cat.cpp
            bmm.cpp
            embedding.cpp
            argmax.cpp
            softmax.cpp
            sort.cpp
	          exponential_.cpp
            reshape_and_cache_flash.cpp
            flash_attn_varlen_func.cpp
            rwkv_mm_sparsity.cpp
            rwkv_ka_fusion.cpp
            copy.cpp
            accuracy_utils.cpp
            xpu_registration.cpp)

if (TRITON_GE_3P5)
  target_compile_definitions(operators PRIVATE TRITON_GE_3P5)
endif()

target_include_directories(operators
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Backend conditional linking
if(FLAGGEMS_BACKEND STREQUAL "CUDA" OR FLAGGEMS_BACKEND STREQUAL "IX")
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit
      PRIVATE CUDA::cudart CUDA::cuda_driver)
elseif(FLAGGEMS_BACKEND STREQUAL "NPU")
    target_link_npu_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
elseif(FLAGGEMS_BACKEND STREQUAL "MUSA")
    target_link_musa_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
elseif(FLAGGEMS_BACKEND STREQUAL "KUNLUNXIN")
    target_link_kunlunxin_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
endif()
target_compile_definitions(operators PRIVATE BACKEND_${FLAGGEMS_BACKEND})

add_library(FlagGems::operators ALIAS operators)


# ==============================================================================
# Pointwise dynamic C++ support (experimental feature, disabled by default)
# ==============================================================================
option(FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP "Build pointwise_dynamic C++ support" OFF)
if(FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP)
  add_subdirectory(pointwise_dynamic_cpp)
  target_sources(operators PRIVATE add.cpp div.cpp fill.cpp)
  target_link_libraries(operators PRIVATE pointwise_dynamic)
  target_compile_definitions(operators PRIVATE FLAGGEMS_POINTWISE_DYNAMIC)
endif()

if (FLAGGEMS_INSTALL)
  include(GNUInstallDirs)
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/flag_gems"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS operators
    EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems
    NAMESPACE FlagGems::
    FILE FlagGemsTargets.cmake)
  export(EXPORT FlagGemsTargets FILE "${PROJECT_BINARY_DIR}/FlagGemsTargets.cmake")
endif()
