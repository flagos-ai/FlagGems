cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(FlagGems LANGUAGES CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ------------------------------- project-wide settings -------------------------------
set(CMAKE_CXX_STANDARD 20) # for fold-expression and concepts
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensures only standard-compliant C++ is used
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------- RPATH settings ---------------------------
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_rpath_portable_origin "@loader_path")
else()
  set(_rpath_portable_origin $ORIGIN)
endif(APPLE)

# default Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# --------------------------- project options ---------------------------
option(FLAGGEMS_USE_EXTERNAL_TRITON_JIT "whether to use external triton jit library" OFF)
option(FLAGGEMS_USE_EXTERNAL_PYBIND11 "whether to use external pybind11 library" ON)
option(FLAGGEMS_BUILD_C_EXTENSIONS "whether to build c extension" OFF)
option(FLAGGEMS_BUILD_CTESTS "Whether to build CPP unit tests" ${FLAGGEMS_BUILD_C_EXTENSIONS})
option(FLAGGEMS_INSTALL "Whether to install the package" ${PROJECT_IS_TOP_LEVEL})

if(NOT FLAGGEMS_BUILD_C_EXTENSIONS)
  return()
endif()

# ==============================================================================
# Backend Selection
# ==============================================================================
set(FLAGGEMS_BACKEND "CUDA" CACHE STRING "Target backend: CUDA, IX, MUSA, NPU")
set_property(CACHE FLAGGEMS_BACKEND PROPERTY STRINGS "CUDA" "IX" "MUSA" "NPU")

if(NOT FLAGGEMS_BACKEND MATCHES "^(CUDA|IX|MUSA|NPU)$")
    message(FATAL_ERROR "Invalid FLAGGEMS_BACKEND: ${FLAGGEMS_BACKEND}")
endif()
message(STATUS "Building FlagGems with backend: ${FLAGGEMS_BACKEND}")

# NPU and MUSA do not need CUDA language
if(FLAGGEMS_BACKEND STREQUAL "NPU" OR FLAGGEMS_BACKEND STREQUAL "MUSA")
    project(FlagGems LANGUAGES CXX VERSION 0.1.0)
else()
    project(FlagGems LANGUAGES CUDA CXX VERSION 0.1.0)
endif()

# ------------------------------- Backend compile definitions -------------------------------
if(FLAGGEMS_BACKEND STREQUAL "CUDA")
    add_compile_definitions(FLAGGEMS_USE_CUDA)
elseif(FLAGGEMS_BACKEND STREQUAL "IX")
    add_compile_definitions(FLAGGEMS_USE_IX)
elseif(FLAGGEMS_BACKEND STREQUAL "NPU")
    add_compile_definitions(FLAGGEMS_USE_NPU)
elseif(FLAGGEMS_BACKEND STREQUAL "MUSA")
    add_compile_definitions(FLAGGEMS_USE_MUSA)
endif()

message(STATUS "Building flag gems with C extensions")
include(FetchContent)

# dependencies: python (must be before backend includes that need Python_EXECUTABLE)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# --------------------------- Backend-specific dependencies ---------------------------
if(FLAGGEMS_BACKEND STREQUAL "CUDA" OR FLAGGEMS_BACKEND STREQUAL "IX")
    find_package(CUDAToolkit REQUIRED)
elseif(FLAGGEMS_BACKEND STREQUAL "NPU")
    include(BackendNPU)
elseif(FLAGGEMS_BACKEND STREQUAL "MUSA")
    include(BackendMUSA)
endif()

# torch
find_package(Torch MODULE REQUIRED) # This is the FindTorch.cmake
# dependencies: json
if(FLAGGEMS_USE_EXTERNAL_TRITON_JIT)
  find_package(TritonJIT 0.1.0 CONFIG REQUIRED)
else()
  set(TRITON_JIT_INSTALL ON) # install triton jit
  set(BACKEND ${FLAGGEMS_BACKEND} CACHE STRING "Backend for TritonJIT" FORCE)
  FetchContent_Declare(TritonJIT
    GIT_REPOSITORY https://github.com/flagos-ai/libtriton_jit
    GIT_TAG master
    # SOURCE_DIR <your local source dir of libtriton_jit>
  )
  FetchContent_MakeAvailable(TritonJIT)
endif()

# dependencies: pybind11
if(FLAGGEMS_USE_EXTERNAL_PYBIND11)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(pybind11 CONFIG REQUIRED)
else()
  FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11)
  FetchContent_MakeAvailable(pybind11)
endif()

if(FLAGGEMS_BUILD_CTESTS)
  set(INSTALL_GTEST OFF) # we do not install tests
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
  )
  FetchContent_MakeAvailable(googletest)
endif()

execute_process(
  #COMMAND ${Python_EXECUTABLE}
  COMMAND ${CMAKE_COMMAND} -E env --unset=LD_PRELOAD ${Python_EXECUTABLE}
          -c "import torch; import triton, sys; sys.stdout.write(triton.__version__)"
  RESULT_VARIABLE _ret
  OUTPUT_VARIABLE TRITON_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _ret EQUAL 0)
  message(FATAL_ERROR "Triton IS NOT installed properly")
endif()
if(TRITON_VERSION VERSION_GREATER_EQUAL "3.5")
  set(TRITON_GE_3P5 1)
endif()
message(STATUS "Detected Triton version: ${TRITON_VERSION}")

# --------------------------- subdirectories ---------------------------
add_subdirectory(lib)
add_subdirectory(src/flag_gems/csrc)
if(FLAGGEMS_BUILD_CTESTS)
  enable_testing()
  add_subdirectory(ctests)
endif()

# --------------------------- install package ---------------------------
if(FLAGGEMS_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/FlagGemsConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FlagGemsConfigVersion.cmake"
    VERSION "${FlagGems_VERSION_MAJOR}.${FlagGems_VERSION_MINOR}.${FlagGems_VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion
  )
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/FlagGemsConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/FlagGemsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems
  )
  # install the FindTorch
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTorch.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems)
endif()
