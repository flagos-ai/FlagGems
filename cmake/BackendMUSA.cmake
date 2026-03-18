# MUSA (Moore Threads) Backend Configuration
message(STATUS "Configuring MUSA backend...")

set(MUSA_HOME $ENV{MUSA_HOME})
if(NOT MUSA_HOME)
    set(MUSA_HOME "/usr/local/musa")
endif()

find_library(MUSA_LIBRARY musa PATHS ${MUSA_HOME}/lib REQUIRED)
find_library(MUSA_RUNTIME_LIBRARY musart PATHS ${MUSA_HOME}/lib REQUIRED)

# Create MUSA::musa_runtime imported target (required by TritonJIT)
if(NOT TARGET MUSA::musa_runtime)
    add_library(MUSA::musa_runtime SHARED IMPORTED)
    set_target_properties(MUSA::musa_runtime PROPERTIES
        IMPORTED_LOCATION "${MUSA_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUSA_HOME}/include"
    )
endif()

# torch_musa path - derive from Python site-packages
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_MUSA_PATH "${PYTHON_SITE_PACKAGES}/torch_musa")
if(NOT EXISTS "${TORCH_MUSA_PATH}")
    set(TORCH_MUSA_PATH "")
endif()
message(STATUS "TORCH_MUSA_PATH: ${TORCH_MUSA_PATH}")

# Find torch_musa library (libmusa_python.so) - provides c10::musa::* symbols
set(TORCH_MUSA_LIB_PATH "${TORCH_MUSA_PATH}/lib")
find_library(TORCH_MUSA_LIBRARY musa_python PATHS ${TORCH_MUSA_LIB_PATH} NO_DEFAULT_PATH)
if(TORCH_MUSA_LIBRARY)
    message(STATUS "Found torch_musa library: ${TORCH_MUSA_LIBRARY}")
else()
    message(WARNING "torch_musa library (libmusa_python.so) not found in ${TORCH_MUSA_LIB_PATH}")
endif()

# Find libittnotify.so (provides iJIT_NotifyEvent symbol)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})
if(CONDA_PREFIX)
    find_library(ITTNOTIFY_LIBRARY ittnotify PATHS ${CONDA_PREFIX}/lib NO_DEFAULT_PATH)
    if(ITTNOTIFY_LIBRARY)
        message(STATUS "Found ittnotify library: ${ITTNOTIFY_LIBRARY}")
    endif()
endif()

# Export libraries for use in other CMakeLists
# Note: libmusa_python.so requires libtorch_python.so for pybind11 symbols
set(MUSA_EXTRA_LIBRARIES "" CACHE INTERNAL "Extra libraries needed for MUSA backend")
if(TORCH_MUSA_LIBRARY)
    list(APPEND MUSA_EXTRA_LIBRARIES ${TORCH_MUSA_LIBRARY})
endif()
if(ITTNOTIFY_LIBRARY)
    list(APPEND MUSA_EXTRA_LIBRARIES ${ITTNOTIFY_LIBRARY})
endif()
# torch_python lib will be added after FindTorch.cmake is included

function(target_link_musa_libraries target)
    target_link_libraries(${target} PRIVATE ${MUSA_LIBRARY})
    target_include_directories(${target} PRIVATE ${MUSA_HOME}/include)
    # Add Python include dirs (needed by torch_musa headers that include pybind11)
    target_include_directories(${target} PRIVATE ${Python_INCLUDE_DIRS})
    if(TORCH_MUSA_PATH)
        # Add torch_musa parent dir for "torch_musa/csrc/..." includes
        get_filename_component(TORCH_MUSA_PARENT "${TORCH_MUSA_PATH}" DIRECTORY)
        target_include_directories(${target} PRIVATE "${TORCH_MUSA_PARENT}")
        target_include_directories(${target} PRIVATE "${TORCH_MUSA_PATH}/include")
    endif()
    # Link torch_musa and ittnotify libraries
    if(TORCH_MUSA_LIBRARY)
        target_link_libraries(${target} PRIVATE ${TORCH_MUSA_LIBRARY})
    endif()
    if(ITTNOTIFY_LIBRARY)
        target_link_libraries(${target} PRIVATE ${ITTNOTIFY_LIBRARY})
    endif()
endfunction()
