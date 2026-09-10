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
# MACA (MetaX) Backend Configuration
# ==============================================================================
message(STATUS "Configuring MACA backend...")

set(MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "Root directory of MACA SDK")
if(NOT MACA_PATH)
    set(MACA_PATH "/opt/maca" CACHE PATH "Root directory of MACA SDK" FORCE)
endif()

set(USE_MACA ON CACHE BOOL "Enable MACA support in dependent CMake packages" FORCE)

if(NOT EXISTS "${MACA_PATH}")
    message(FATAL_ERROR "MACA SDK not found at ${MACA_PATH}. Please set MACA_PATH.")
endif()
message(STATUS "MACA_PATH: ${MACA_PATH}")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17 CACHE STRING "CUDA standard used by MACA cu-bridge" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_STANDARD_REQUIRED)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "Require the selected CUDA standard" FORCE)
endif()

find_path(MACA_INCLUDE_DIR
    NAMES mcr/mc_runtime.h
    HINTS "${MACA_PATH}"
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    REQUIRED
)

find_path(MACA_CU_BRIDGE_INCLUDE_DIR
    NAMES cuda_runtime_api.h cuda.h
    HINTS "${MACA_PATH}/tools/cu-bridge"
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(MACA_RUNTIME_LIBRARY
    NAMES mcruntime
    HINTS "${MACA_PATH}"
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
    REQUIRED
)

message(STATUS "Found MACA include: ${MACA_INCLUDE_DIR}")
message(STATUS "Found MACA cu-bridge include: ${MACA_CU_BRIDGE_INCLUDE_DIR}")
message(STATUS "Found MACA runtime: ${MACA_RUNTIME_LIBRARY}")

if(NOT TARGET MACA::mcruntime)
    add_library(MACA::mcruntime SHARED IMPORTED)
    set_target_properties(MACA::mcruntime PROPERTIES
        IMPORTED_LOCATION "${MACA_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MACA_INCLUDE_DIR};${MACA_INCLUDE_DIR}/mcr"
    )
endif()

list(APPEND CMAKE_INSTALL_RPATH "${MACA_PATH}/lib" "${MACA_PATH}/mxgpu_llvm/lib")

function(target_link_maca_libraries target)
    target_link_libraries(${target} PRIVATE MACA::mcruntime)
    target_include_directories(${target} BEFORE PUBLIC ${MACA_CU_BRIDGE_INCLUDE_DIR} ${MACA_INCLUDE_DIR})
    target_compile_definitions(${target} PUBLIC USE_MACA)
endfunction()

message(STATUS "MACA backend configuration complete")
