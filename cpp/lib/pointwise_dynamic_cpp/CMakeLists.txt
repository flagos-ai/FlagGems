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

# ==============================================================================
# CMakeLists.txt for pointwise_dynamic C++ implementation
# ==============================================================================

set(POINTWISE_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
# Export the generated dir to the parent (lib/) scope so operators can compile
# the generated glue body (pointwise_ops_glue.cc) that lands here.
set(POINTWISE_GENERATED_DIR "${POINTWISE_OUTPUT_DIR}" PARENT_SCOPE)

# Operations to generate (full module paths)
set(POINTWISE_OP_FILES "flag_gems.ops.add" "flag_gems.ops.div" "flag_gems.ops.fill"
    "flag_gems.ops.remainder"
    "flag_gems.ops.abs" "flag_gems.ops.neg" "flag_gems.ops.exp" "flag_gems.ops.sqrt"
    "flag_gems.ops.rsqrt" "flag_gems.ops.tanh" "flag_gems.ops.sigmoid" "flag_gems.ops.silu"
    "flag_gems.ops.relu" "flag_gems.ops.gelu"
    CACHE STRING "Space-separated list of op module paths" FORCE)

# Maximum tensor rank
set(POINTWISE_MAX_RANK 5 CACHE STRING "Maximum tensor rank")

# When ON, all pointwise ops are registered through a single generic boxed
# adapter (flaggems_pointwise_boxed) driven by the generated ATEN_TO_BOXED
# routing table, instead of per-op unboxed TORCH_FN glue. Adding a new
# pointwise op then costs one line in op_specs.py and zero hand-written C++.
option(FLAGGEMS_POINTWISE_DYNAMIC_BOXED
    "Use the generic boxed dispatch adapter for pointwise ops" OFF)

set(POINTWISE_BOXED_FLAG "")
if(FLAGGEMS_POINTWISE_DYNAMIC_BOXED)
    set(POINTWISE_BOXED_FLAG "--boxed")
endif()

# ==============================================================================
# Generate kernels at BUILD TIME (not configure time)
# ==============================================================================

file(MAKE_DIRECTORY ${POINTWISE_OUTPUT_DIR})

set(PREBUILD_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/prebuild_kernels.py)

set(POINTWISE_MANIFEST_H ${POINTWISE_OUTPUT_DIR}/pointwise_manifest.h)
set(POINTWISE_RUNTIME_H ${POINTWISE_OUTPUT_DIR}/pointwise_runtime.h)
set(POINTWISE_REPORT ${POINTWISE_OUTPUT_DIR}/kernel_report.txt)
# aten-facing glue generated from op_specs.py (declarations, passthrough/selector
# bodies, and the three cstub registration blocks).
set(POINTWISE_OPS_GLUE_H ${POINTWISE_OUTPUT_DIR}/pointwise_ops_glue.h)
set(POINTWISE_OPS_GLUE_CC ${POINTWISE_OUTPUT_DIR}/pointwise_ops_glue.cc)
set(POINTWISE_CSTUB_INC ${POINTWISE_OUTPUT_DIR}/pointwise_cstub.inc)
# Boxed routing table (BoxedRecipe + ATEN_TO_BOXED). Always generated; consumed
# by pointwise_boxed.h only under FLAGGEMS_POINTWISE_DYNAMIC_BOXED.
set(POINTWISE_ATEN_TO_KERNEL_H ${POINTWISE_OUTPUT_DIR}/pointwise_aten_to_kernel.h)

# Always regenerate kernels at build time.
# This ensures kernel .py cache files are recreated even if the
# C++ headers already exist (e.g. after clearing the code cache).
add_custom_target(pointwise_generate_kernels ALL
    COMMAND ${Python_EXECUTABLE}
        ${PREBUILD_SCRIPT}
        --op-files ${POINTWISE_OP_FILES}
        --max-rank ${POINTWISE_MAX_RANK}
        --output-dir ${POINTWISE_OUTPUT_DIR}
        ${POINTWISE_BOXED_FLAG}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating pointwise kernels (build-time, shared codegen)..."
    BYPRODUCTS
        ${POINTWISE_MANIFEST_H}
        ${POINTWISE_RUNTIME_H}
        ${POINTWISE_REPORT}
        ${POINTWISE_OPS_GLUE_H}
        ${POINTWISE_OPS_GLUE_CC}
        ${POINTWISE_CSTUB_INC}
        ${POINTWISE_ATEN_TO_KERNEL_H}
)

# ==============================================================================
# Create interface library
# ==============================================================================
add_library(pointwise_dynamic INTERFACE)

target_include_directories(pointwise_dynamic
    INTERFACE
        $<BUILD_INTERFACE:${POINTWISE_OUTPUT_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/flag_gems/pointwise_dynamic>
)

# Ensure headers are generated before anything uses this library
add_dependencies(pointwise_dynamic pointwise_generate_kernels)

# ==============================================================================
# Install
# ==============================================================================
if(FLAGGEMS_INSTALL)
    install(FILES
        ${POINTWISE_MANIFEST_H}
        ${POINTWISE_RUNTIME_H}
        ${POINTWISE_OPS_GLUE_H}
        ${POINTWISE_ATEN_TO_KERNEL_H}
        ${CMAKE_CURRENT_SOURCE_DIR}/pointwise_boxed.h
        DESTINATION include/flag_gems/pointwise_dynamic
    )
endif()

# ==============================================================================
# Status message (configure time)
# ==============================================================================
message(STATUS "Pointwise dynamic C++ support configured:")
message(STATUS "  Op files: ${POINTWISE_OP_FILES}")
message(STATUS "  Max rank: ${POINTWISE_MAX_RANK}")
message(STATUS "  Output dir: ${POINTWISE_OUTPUT_DIR}")
message(STATUS "  Boxed dispatch: ${FLAGGEMS_POINTWISE_DYNAMIC_BOXED}")
message(STATUS "  Kernels will be generated at BUILD time (not configure time)")
