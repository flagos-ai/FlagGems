# ==============================================================================
# Kunlunxin (Baidu XPU3) Backend Configuration
# ==============================================================================
message(STATUS "Configuring Kunlunxin backend...")

# Derive conda env root (Kunlunxin XRE3 SDK is often shipped inside a conda env,
# e.g. $CONDA_PREFIX/xcudart for the xre-xcn bundle; we also honor XRE3_HOME / XPU_HOME).
get_filename_component(PYTHON_BIN_DIR ${Python_EXECUTABLE} DIRECTORY)
get_filename_component(CONDA_ENV_ROOT ${PYTHON_BIN_DIR} DIRECTORY)

# ------------------------------- Find XRE3 (xpu runtime) ----------------------
# xpu/runtime.h provides: XPUStream, xpu_current_device, xpu_stream_create, ...
find_path(KUNLUNXIN_RUNTIME_INCLUDE_DIR
    NAMES xpu/runtime.h
    HINTS
        ENV XRE3_HOME
        ENV XPU_HOME
        ENV XPU_SDK_ROOT
        ${CONDA_ENV_ROOT}/xcudart
        ${CONDA_ENV_ROOT}
    PATH_SUFFIXES include
    PATHS /usr/include /usr/local/include
)

find_library(KUNLUNXIN_RUNTIME_LIBRARY
    NAMES xpurt xpu_runtime
    HINTS
        ENV XRE3_HOME
        ENV XPU_HOME
        ENV XPU_SDK_ROOT
        ${CONDA_ENV_ROOT}/xcudart
        ${CONDA_ENV_ROOT}
    PATH_SUFFIXES so lib lib64
    PATHS /usr/lib /usr/local/lib
)

if(NOT KUNLUNXIN_RUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "Kunlunxin runtime headers (xpu/runtime.h) not found. "
                        "Set XRE3_HOME or XPU_HOME environment variable.")
endif()
if(NOT KUNLUNXIN_RUNTIME_LIBRARY)
    message(FATAL_ERROR "Kunlunxin runtime library (libxpurt) not found. "
                        "Set XRE3_HOME or XPU_HOME environment variable.")
endif()
message(STATUS "Found Kunlunxin runtime headers: ${KUNLUNXIN_RUNTIME_INCLUDE_DIR}")
message(STATUS "Found Kunlunxin runtime library: ${KUNLUNXIN_RUNTIME_LIBRARY}")

# ------------------------------- Find XDNN (xpu/xdnn.h) -----------------------
# xpu/xdnn.h is pulled in transitively by torch_xmlir/runtime/xpu_context.h.
# The XDNN API SDK is a *separate* package from the XRE3 runtime.
find_path(KUNLUNXIN_XDNN_INCLUDE_DIR
    NAMES xpu/xdnn.h
    HINTS
        ENV XDNN_HOME
        ENV XPU_XDNN_HOME
        ENV XPU_API_HOME
        ENV XPU_HOME
        ${CONDA_ENV_ROOT}
    PATH_SUFFIXES include output/include api/output/include
    PATHS /usr/include /usr/local/include
)
find_library(KUNLUNXIN_XDNN_LIBRARY
    NAMES xpuapi
    HINTS
        ENV XDNN_HOME
        ENV XPU_XDNN_HOME
        ENV XPU_API_HOME
        ENV XPU_HOME
        ${CONDA_ENV_ROOT}
    PATH_SUFFIXES so lib lib64 output/so api/output/so
    PATHS /usr/lib /usr/local/lib
)
if(NOT KUNLUNXIN_XDNN_INCLUDE_DIR)
    message(FATAL_ERROR "Kunlunxin XDNN headers (xpu/xdnn.h) not found. "
                        "Set XDNN_HOME (e.g. /path/to/xpu/api/output) environment variable.")
endif()
message(STATUS "Found Kunlunxin XDNN headers: ${KUNLUNXIN_XDNN_INCLUDE_DIR}")
if(KUNLUNXIN_XDNN_LIBRARY)
    message(STATUS "Found Kunlunxin XDNN library: ${KUNLUNXIN_XDNN_LIBRARY}")
else()
    message(STATUS "Kunlunxin XDNN library (libxpuapi) not found; headers only (OK if unused).")
endif()

# ------------------------------- Find BKCL (bkcl.h) ---------------------------
# torch_xmlir/runtime/xpu_context.h pulls in bkcl.h. When WITHOUT_MLIR is
# defined it uses ``#include "bkcl.h"`` (flat), otherwise ``xpu/bkcl.h``.
# The BKCL header ships inside the xcn_torch / xtrans_cuda bundle as a flat
# ``<bkcl.h>``, so we unconditionally define WITHOUT_MLIR and expose that path.
find_path(KUNLUNXIN_BKCL_INCLUDE_DIR
    NAMES bkcl.h
    HINTS
        ENV BKCL_HOME
        ENV XPU_BKCL_HOME
        ENV XPU_HOME
    PATH_SUFFIXES include targets/x86_64-linux/include
    PATHS /usr/include /usr/local/include
)
if(NOT KUNLUNXIN_BKCL_INCLUDE_DIR)
    message(FATAL_ERROR "Kunlunxin BKCL header (bkcl.h) not found. "
                        "Set BKCL_HOME (e.g. /path/to/xtrans_cuda_.../targets/x86_64-linux).")
endif()
message(STATUS "Found Kunlunxin BKCL header: ${KUNLUNXIN_BKCL_INCLUDE_DIR}")

# ------------------------------- torch_xmlir Integration ----------------------
# torch_xmlir registers the PrivateUse1 backend for Baidu KL-series devices and
# provides ``xmlir_rt::getCurrentStream()`` which returns an ``XPUStream``
# (aka ``void*``) suitable for passing to triton_jit's Kunlunxin backend.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch_xmlir, os; print(os.path.dirname(torch_xmlir.__file__))"
    OUTPUT_VARIABLE TORCH_XMLIR_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _torch_xmlir_ret
)

set(TORCH_XMLIR_INCLUDE_DIR "")
set(TORCH_XMLIR_RUNTIME_LIBRARY "")
if(_torch_xmlir_ret EQUAL 0 AND TORCH_XMLIR_PATH)
    message(STATUS "Found torch_xmlir at: ${TORCH_XMLIR_PATH}")
    set(TORCH_XMLIR_INCLUDE_DIR "${TORCH_XMLIR_PATH}/include")
    find_library(TORCH_XMLIR_RUNTIME_LIBRARY
        NAMES XMLIRRuntime
        PATHS ${TORCH_XMLIR_PATH}
        NO_DEFAULT_PATH
    )
    if(TORCH_XMLIR_RUNTIME_LIBRARY)
        message(STATUS "Found libXMLIRRuntime: ${TORCH_XMLIR_RUNTIME_LIBRARY}")
    else()
        message(WARNING "torch_xmlir found but libXMLIRRuntime.so missing at ${TORCH_XMLIR_PATH}; "
                        "getCurrentStream() will not resolve.")
    endif()
else()
    message(WARNING "torch_xmlir not importable; KL3 stream integration may fail.")
endif()

# ------------------------------- IMPORTED target ------------------------------
# Collect all include dirs so every consumer of Kunlunxin::runtime sees them.
set(_KUNLUNXIN_INTERFACE_INCLUDES
    "${KUNLUNXIN_RUNTIME_INCLUDE_DIR}"
    "${KUNLUNXIN_XDNN_INCLUDE_DIR}"
    "${KUNLUNXIN_BKCL_INCLUDE_DIR}")
if(TORCH_XMLIR_INCLUDE_DIR)
    list(APPEND _KUNLUNXIN_INTERFACE_INCLUDES "${TORCH_XMLIR_INCLUDE_DIR}")
endif()

if(NOT TARGET Kunlunxin::runtime)
    add_library(Kunlunxin::runtime UNKNOWN IMPORTED)
    set_target_properties(Kunlunxin::runtime PROPERTIES
        IMPORTED_LOCATION "${KUNLUNXIN_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${_KUNLUNXIN_INTERFACE_INCLUDES}"
        INTERFACE_COMPILE_DEFINITIONS "WITHOUT_MLIR"
    )
    if(TORCH_XMLIR_RUNTIME_LIBRARY)
        set_target_properties(Kunlunxin::runtime PROPERTIES
            INTERFACE_LINK_LIBRARIES "${TORCH_XMLIR_RUNTIME_LIBRARY}")
    endif()
endif()

# ------------------------------- Helper Function ------------------------------
# PUBLIC linkage so transitively any ctest/binary linking `operators` picks up
# the xpu/runtime.h + xpu/xdnn.h + torch_xmlir include paths.
function(target_link_kunlunxin_libraries target)
    target_link_libraries(${target} PUBLIC Kunlunxin::runtime)
endfunction()

message(STATUS "Kunlunxin backend configuration complete")
