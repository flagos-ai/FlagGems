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
# HCU (Hygon) Backend Configuration
# ==============================================================================
message(STATUS "Configuring HCU backend...")

# HCU uses HIP runtime. Find HIP package — provided by DTK (Hygon Developer ToolKit).
#
# Users can set the DTK_PATH environment variable to point to DTK installation:
#   export DTK_PATH=/opt/dtk-26.04
#
# If DTK_PATH is not set, the build will fall back to common DTK installation
# paths (/opt/dtk, /opt/dtk-*) and ROCm-compatible paths for flexibility.
if(DEFINED ENV{DTK_PATH})
    list(APPEND CMAKE_PREFIX_PATH
        "$ENV{DTK_PATH}"
        "$ENV{DTK_PATH}/hip"
    )
    message(STATUS "Using DTK_PATH (env): $ENV{DTK_PATH}")
elseif(DEFINED DTK_PATH)
    list(APPEND CMAKE_PREFIX_PATH
        "${DTK_PATH}"
        "${DTK_PATH}/hip"
    )
    message(STATUS "Using DTK_PATH (cmake var): ${DTK_PATH}")
else()
    list(APPEND CMAKE_PREFIX_PATH
        "$ENV{ROCM_PATH}"
        "$ENV{HIP_PATH}"
        "/opt/dtk"
        "/opt/dtk/hip"
    )
    # Glob for versioned DTK directories (e.g. /opt/dtk-26.04) when no
    # symlink /opt/dtk is present.
    file(GLOB _dtk_versions "/opt/dtk-*")
    if(_dtk_versions)
        list(APPEND CMAKE_PREFIX_PATH ${_dtk_versions})
        foreach(_v ${_dtk_versions})
            list(APPEND CMAKE_PREFIX_PATH "${_v}/hip")
        endforeach()
    endif()
    message(STATUS "DTK_PATH not set, searching default paths")
endif()

find_package(hip QUIET)
if(NOT hip_FOUND)
    message(FATAL_ERROR
        "HIP (find_package(hip)) not found. "
        "Please set DTK_PATH to your DTK installation, e.g.\n"
        "  export DTK_PATH=/opt/dtk-26.04\n"
        "or source the DTK environment:\n"
        "  source /opt/dtk-26.04/env.sh")
endif()
message(STATUS "Found HIP: ${hip_VERSION}")

# PyTorch pip wheels bundle libibverbs inside torch.libs/ with hashed SONAMEs
# that are not on the linker search path. Suppress the link-time error.
add_link_options(-Wl,--allow-shlib-undefined)

# ------------------------------- Helper Function ------------------------------
function(target_link_hcu_libraries target_name)
    target_link_libraries(${target_name} PRIVATE hip::host)
    # torch_hip sets -std=c++17 in INTERFACE_COMPILE_OPTIONS, which overrides our
    # C++20 standard. Strip it so C++20 concepts work.
    # Note: torch_hip target is created by find_package(Torch), so it exists here.
    if(TARGET torch_hip)
        get_target_property(_torch_hip_opts torch_hip INTERFACE_COMPILE_OPTIONS)
        if(_torch_hip_opts)
            list(FILTER _torch_hip_opts EXCLUDE REGEX "-std=c\\+\\+17")
            list(FILTER _torch_hip_opts EXCLUDE REGEX "-Wno-duplicate-decl-specifier")
            set_property(TARGET torch_hip PROPERTY INTERFACE_COMPILE_OPTIONS ${_torch_hip_opts})
        endif()
    endif()
endfunction()

message(STATUS "HCU backend configuration complete")

# Create librt.so symlink if missing (Ubuntu 22.04+ merged librt into libc)
if(NOT EXISTS "/usr/lib/x86_64-linux-gnu/librt.so"
   AND EXISTS "/usr/lib/x86_64-linux-gnu/librt.so.1")
    message(STATUS "librt.so stub missing — creating symlink")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            /usr/lib/x86_64-linux-gnu/librt.so.1
            /usr/lib/x86_64-linux-gnu/librt.so
    )
endif()
