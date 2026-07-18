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
# Enflame GCU Backend Configuration
# ==============================================================================
message(STATUS "Configuring GCU (Enflame) backend...")

set(TOPS_ROOT "/opt/tops" CACHE PATH "Root directory of TOPS SDK")

find_path(GCU_RUNTIME_INCLUDE_DIR
    NAMES tops/tops_runtime.h tops_runtime_api.h
    HINTS
        ${TOPS_ROOT}
        ENV TOPS_ROOT
    PATH_SUFFIXES include
    PATHS /opt/tops/include /usr/include /usr/local/include
)

find_library(GCU_RUNTIME_LIBRARY
    NAMES topsrt
    HINTS
        ${TOPS_ROOT}
        ENV TOPS_ROOT
    PATH_SUFFIXES lib lib64
    PATHS /opt/tops/lib /usr/lib /usr/local/lib
)

if(NOT GCU_RUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "GCU runtime headers (tops_runtime_api.h) not found. "
                        "Set TOPS_ROOT or install TOPS SDK to /opt/tops.")
endif()
if(NOT GCU_RUNTIME_LIBRARY)
    message(FATAL_ERROR "GCU runtime library (libtopsrt.so) not found. "
                        "Set TOPS_ROOT or install TOPS SDK to /opt/tops.")
endif()
message(STATUS "Found GCU runtime headers: ${GCU_RUNTIME_INCLUDE_DIR}")
message(STATUS "Found GCU runtime library: ${GCU_RUNTIME_LIBRARY}")

if(NOT TARGET GCU::efrt)
    add_library(GCU::efrt SHARED IMPORTED)
    set_target_properties(GCU::efrt PROPERTIES
        IMPORTED_LOCATION "${GCU_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GCU_RUNTIME_INCLUDE_DIR}"
    )
endif()

function(target_link_gcu_libraries target)
    target_include_directories(${target} PUBLIC ${GCU_RUNTIME_INCLUDE_DIR})
    target_link_libraries(${target} PUBLIC GCU::efrt)
endfunction()

message(STATUS "GCU backend configuration complete")
