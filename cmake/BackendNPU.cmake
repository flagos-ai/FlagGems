# ==============================================================================
# NPU (Ascend) Backend Configuration
# ==============================================================================
message(STATUS "Configuring NPU backend...")

# ------------------------------- Ascend Toolkit -------------------------------
set(ASCEND_HOME $ENV{ASCEND_TOOLKIT_HOME})
if(NOT ASCEND_HOME)
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

if(NOT EXISTS ${ASCEND_HOME})
    message(FATAL_ERROR "Ascend toolkit not found at ${ASCEND_HOME}. "
                        "Please set ASCEND_TOOLKIT_HOME environment variable.")
endif()
message(STATUS "ASCEND_TOOLKIT_HOME: ${ASCEND_HOME}")

# Find CANN installation for runtime headers (rt.h)
set(CANN_HOME $ENV{CANN_HOME})
if(NOT CANN_HOME)
    file(GLOB CANN_CANDIDATES "/usr/local/Ascend/cann-*")
    if(CANN_CANDIDATES)
        list(GET CANN_CANDIDATES 0 CANN_HOME)
    endif()
endif()

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ASCEND_ARCH_DIR "aarch64-linux")
else()
    set(ASCEND_ARCH_DIR "x86_64-linux")
endif()

# ------------------------------- Find Ascend Libraries ------------------------
find_library(ASCENDCL_LIBRARY ascendcl PATHS ${ASCEND_HOME}/lib64 NO_DEFAULT_PATH REQUIRED)
find_library(ASCEND_RUNTIME_LIBRARY runtime PATHS ${ASCEND_HOME}/lib64 NO_DEFAULT_PATH REQUIRED)

message(STATUS "Found AscendCL: ${ASCENDCL_LIBRARY}")
message(STATUS "Found Ascend Runtime: ${ASCEND_RUNTIME_LIBRARY}")

# ------------------------------- Ascend Include Directories -------------------
set(ASCEND_INCLUDE_DIRS
    "${ASCEND_HOME}/include"
    "${ASCEND_HOME}/include/aclnn"
    "${ASCEND_HOME}/include/experiment"
    "${ASCEND_HOME}/include/experiment/runtime"
    "${ASCEND_HOME}/${ASCEND_ARCH_DIR}/include"
    "${ASCEND_HOME}/${ASCEND_ARCH_DIR}/include/experiment"
    "${ASCEND_HOME}/${ASCEND_ARCH_DIR}/include/experiment/msprof"
)

if(CANN_HOME AND EXISTS "${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc")
    list(APPEND ASCEND_INCLUDE_DIRS "${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc")
    message(STATUS "Found CANN pkg_inc: ${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc")
    if(EXISTS "${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc/runtime/runtime")
        list(APPEND ASCEND_INCLUDE_DIRS "${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc/runtime/runtime")
    endif()
endif()

# ------------------------------- Create Imported Targets ----------------------
# These targets are required by TritonJIT (fetched via FetchContent).
# Guard with if(NOT TARGET ...) to avoid duplicate definition when
# TritonJIT's own BackendNPU.cmake is also included.
if(NOT TARGET Ascend::ascendcl)
    add_library(Ascend::ascendcl SHARED IMPORTED)
    set_target_properties(Ascend::ascendcl PROPERTIES
        IMPORTED_LOCATION ${ASCENDCL_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_INCLUDE_DIRS}"
    )
endif()

if(NOT TARGET Ascend::runtime)
    add_library(Ascend::runtime SHARED IMPORTED)
    set_target_properties(Ascend::runtime PROPERTIES
        IMPORTED_LOCATION ${ASCEND_RUNTIME_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_INCLUDE_DIRS}"
    )
endif()

# ------------------------------- torch_npu Integration ------------------------
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch_npu; print(torch_npu.__path__[0])"
    OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)

if(TORCH_NPU_PATH)
    message(STATUS "Found torch_npu at: ${TORCH_NPU_PATH}")
    find_library(TORCH_NPU_LIB torch_npu
        PATHS "${TORCH_NPU_PATH}/lib"
        NO_DEFAULT_PATH
    )
    if(TORCH_NPU_LIB)
        message(STATUS "Found torch_npu library: ${TORCH_NPU_LIB}")
    else()
        message(WARNING "torch_npu package found but libtorch_npu.so not found in ${TORCH_NPU_PATH}/lib")
    endif()
else()
    message(WARNING "torch_npu not found via Python import")
endif()

# ------------------------------- Helper Function ------------------------------
function(target_link_npu_libraries target)
    target_link_libraries(${target} PRIVATE Ascend::ascendcl Ascend::runtime)
    target_include_directories(${target} PRIVATE ${ASCEND_INCLUDE_DIRS})
    if(TORCH_NPU_PATH)
        target_include_directories(${target} PRIVATE "${TORCH_NPU_PATH}/include")
    endif()
    if(TORCH_NPU_LIB)
        target_link_libraries(${target} PRIVATE ${TORCH_NPU_LIB})
    endif()
endfunction()

message(STATUS "NPU backend configuration complete")
