# ==============================================================================
# CMakeLists.txt for pointwise_dynamic C++ implementation
# ==============================================================================

set(POINTWISE_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Operations to generate (full module paths)
set(POINTWISE_OP_FILES "flag_gems.ops.add"
    CACHE STRING "Comma-separated list of op module paths")

# Maximum tensor rank
set(POINTWISE_MAX_RANK 5 CACHE STRING "Maximum tensor rank")

# ==============================================================================
# Generate kernels at BUILD TIME (not configure time)
# ==============================================================================

file(MAKE_DIRECTORY ${POINTWISE_OUTPUT_DIR})

set(PREBUILD_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/prebuild_kernels.py)

set(POINTWISE_MANIFEST_H ${POINTWISE_OUTPUT_DIR}/pointwise_manifest.h)
set(POINTWISE_RUNTIME_H ${POINTWISE_OUTPUT_DIR}/pointwise_runtime.h)
set(POINTWISE_REPORT ${POINTWISE_OUTPUT_DIR}/kernel_report.txt)

# Always regenerate kernels at build time.
# This ensures kernel .py cache files are recreated even if the
# C++ headers already exist (e.g. after clearing the code cache).
add_custom_target(pointwise_generate_kernels ALL
    COMMAND ${Python_EXECUTABLE}
        ${PREBUILD_SCRIPT}
        --op-files ${POINTWISE_OP_FILES}
        --max-rank ${POINTWISE_MAX_RANK}
        --output-dir ${POINTWISE_OUTPUT_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating pointwise kernels (build-time, shared codegen)..."
    BYPRODUCTS
        ${POINTWISE_MANIFEST_H}
        ${POINTWISE_RUNTIME_H}
        ${POINTWISE_REPORT}
)

# ==============================================================================
# Create interface library
# ==============================================================================
add_library(pointwise_dynamic INTERFACE)

target_include_directories(pointwise_dynamic
    INTERFACE
        $<BUILD_INTERFACE:${POINTWISE_OUTPUT_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
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
message(STATUS "  Kernels will be generated at BUILD time (not configure time)")
