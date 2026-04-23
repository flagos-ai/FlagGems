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

# ------------------------------- IMPORTED target ------------------------------
if(NOT TARGET Kunlunxin::runtime)
    add_library(Kunlunxin::runtime UNKNOWN IMPORTED)
    set_target_properties(Kunlunxin::runtime PROPERTIES
        IMPORTED_LOCATION "${KUNLUNXIN_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${KUNLUNXIN_RUNTIME_INCLUDE_DIR}"
    )
endif()

# ------------------------------- Helper Function ------------------------------
function(target_link_kunlunxin_libraries target)
    target_link_libraries(${target} PUBLIC Kunlunxin::runtime)
endfunction()

message(STATUS "Kunlunxin backend configuration complete")
