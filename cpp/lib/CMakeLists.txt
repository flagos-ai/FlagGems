# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(operators
            SHARED
            device_info.cpp
            act_quant.cpp
            zeros.cpp
            utils.cpp
            sum.cpp
            max.cpp
            mm.cpp
            rms_norm.cpp
            gemma_rms_norm.cpp
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
            fp8_matmul.cpp
            fp8_matmul_direct.cpp)

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
elseif(FLAGGEMS_BACKEND STREQUAL "GCU")
    target_link_gcu_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
elseif(FLAGGEMS_BACKEND STREQUAL HCU)
    target_link_hcu_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
    target_compile_options(operators PRIVATE -std=c++20)
elseif(FLAGGEMS_BACKEND STREQUAL "MACA")
    target_link_maca_libraries(operators)
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
  # Hand-written bodies: add/div/fill/remainder families (complex dispatch,
  # rounding modes, Scalar->double, 0-dim branching). The passthrough/selector
  # bodies (unary + gelu) are auto-generated into pointwise_ops_glue.cc by
  # prebuild_kernels.py from op_specs.py.
  target_sources(operators PRIVATE add.cpp div.cpp fill.cpp
    "${POINTWISE_GENERATED_DIR}/pointwise_ops_glue.cc")
  set_source_files_properties(
    "${POINTWISE_GENERATED_DIR}/pointwise_ops_glue.cc"
    PROPERTIES GENERATED TRUE)
  target_link_libraries(operators PRIVATE pointwise_dynamic)
  add_dependencies(operators pointwise_generate_kernels)
  # Expose the generated include dir PUBLICly (BUILD_INTERFACE only -- nothing to
  # install beyond the headers already handled by pointwise_dynamic_cpp) so that
  # pointwise_ops_glue.h, included transitively by operators.h, resolves for
  # consumers (c_operators, ctests). The pointwise_dynamic INTERFACE lib itself
  # stays PRIVATE to keep it out of the operators export set.
  target_include_directories(operators PUBLIC
    $<BUILD_INTERFACE:${POINTWISE_GENERATED_DIR}>)
  target_compile_definitions(operators PRIVATE FLAGGEMS_POINTWISE_DYNAMIC)

  if(FLAGGEMS_POINTWISE_DYNAMIC_BOXED)
    # The boxed adapter's DEFINITION (pointwise_boxed.cc, which #defines
    # FLAGGEMS_POINTWISE_BOXED_IMPL) is compiled into `operators` -- the only
    # target with the CUDA / triton_jit include paths that
    # pointwise_prepare_args.h transitively needs.
    target_sources(operators PRIVATE
      pointwise_dynamic_cpp/pointwise_boxed.cc)
    # cstub.cpp includes the DECLARATION-only pointwise_boxed.h (no CUDA deps),
    # so the source dir must be on the PUBLIC include path.
    target_include_directories(operators PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pointwise_dynamic_cpp>)
    # The registration switch must reach both operators and its consumer
    # (c_operators) so the generated boxed m.impl lines are compiled in.
    target_compile_definitions(operators PUBLIC FLAGGEMS_POINTWISE_DYNAMIC_BOXED)
  endif()
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
