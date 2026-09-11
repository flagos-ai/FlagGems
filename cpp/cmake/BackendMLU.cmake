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
# MLU (Cambricon) Backend Configuration
# ==============================================================================
message(STATUS "Configuring MLU backend...")

# ------------------------------- Neuware SDK ----------------------------------
set(NEUWARE_HOME $ENV{NEUWARE_HOME})
if(NOT NEUWARE_HOME)
    set(NEUWARE_HOME "/usr/local/neuware")
endif()

if(NOT EXISTS "${NEUWARE_HOME}/include/cn_api.h")
    message(FATAL_ERROR "Neuware SDK not found at ${NEUWARE_HOME}/include/cn_api.h. "
                        "Please set NEUWARE_HOME environment variable.")
endif()
set(ENV{NEUWARE_HOME} "${NEUWARE_HOME}")
message(STATUS "NEUWARE_HOME: ${NEUWARE_HOME}")

# ------------------------------- torch_mlu Integration ------------------------
# torch_mlu.utils.cmake_prefix_path reports <torch_mlu>/share/cmake, but the
# package config actually lives under <torch_mlu>/csrc/share/cmake/TorchMLU.
# Probe the reported prefix first and fall back to the real location.
if(NOT DEFINED TorchMLU_DIR AND NOT DEFINED TorchMLU_ROOT)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import os, torch_mlu; print(os.path.dirname(torch_mlu.__file__))"
        OUTPUT_VARIABLE TORCH_MLU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch_mlu; print(torch_mlu.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TorchMLU_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if(NOT EXISTS "${TorchMLU_ROOT}/TorchMLU/TorchMLUConfig.cmake"
       AND NOT EXISTS "${TorchMLU_ROOT}/TorchMLUConfig.cmake"
       AND EXISTS "${TORCH_MLU_PATH}/csrc/share/cmake/TorchMLU/TorchMLUConfig.cmake")
        set(TorchMLU_DIR "${TORCH_MLU_PATH}/csrc/share/cmake/TorchMLU")
        message(STATUS "TorchMLUConfig.cmake not under the reported prefix; "
                       "falling back to TorchMLU_DIR=${TorchMLU_DIR}")
    endif()
endif()

find_package(TorchMLU CONFIG REQUIRED)
message(STATUS "Found MLU Runtime (TorchMLU)")

# TorchMLUConfig.cmake resets CMAKE_MODULE_PATH; restore this directory so the
# subsequent find_package(Torch MODULE) can still locate FindTorch.cmake.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

# TorchMLU exports include dirs/libraries that do not always exist in every
# torch_mlu build (e.g. libaten_mlu.so is merged into libtorch_mlu.so in
# recent releases, leaving a -NOTFOUND entry). Filter those out.
set(MLU_INCLUDE_DIRS "")
foreach(_dir IN LISTS TORCH_MLU_INCLUDE_DIRS)
    if(IS_DIRECTORY "${_dir}")
        list(APPEND MLU_INCLUDE_DIRS "${_dir}")
    else()
        message(STATUS "Skipping non-existent MLU include dir: ${_dir}")
    endif()
endforeach()
list(APPEND MLU_INCLUDE_DIRS "${NEUWARE_HOME}/include")

set(MLU_LIBRARIES "")
foreach(_lib IN LISTS TORCH_MLU_LIBRARIES)
    if(_lib MATCHES "-NOTFOUND$")
        message(STATUS "Skipping missing MLU library: ${_lib}")
    else()
        list(APPEND MLU_LIBRARIES "${_lib}")
    endif()
endforeach()

message(STATUS "MLU include dirs: ${MLU_INCLUDE_DIRS}")
message(STATUS "MLU libraries: ${MLU_LIBRARIES}")

# Apply globally so translation units that include backend_utils.h (which
# references cnrt.h and torch_mlu's framework headers) resolve headers even
# when not explicitly linked via target_link_mlu_libraries().
include_directories(${MLU_INCLUDE_DIRS})

# ------------------------------- Create Imported Targets ----------------------
# Guard with if(NOT TARGET ...) to avoid duplicate definition when TritonJIT's
# own BackendMLU.cmake is also included.
if(NOT TARGET MLU::mlu_runtime)
    add_library(MLU::mlu_runtime INTERFACE IMPORTED)
    set_target_properties(MLU::mlu_runtime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MLU_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${MLU_LIBRARIES}"
    )
endif()

# ------------------------------- Helper Function ------------------------------
function(target_link_mlu_libraries target)
    target_include_directories(${target} PRIVATE ${MLU_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${MLU_LIBRARIES})
endfunction()

message(STATUS "MLU backend configuration complete")
