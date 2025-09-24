cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(FlagGems LANGUAGES CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ------------------------------- project-wide settings -------------------------------
set(CMAKE_CXX_STANDARD 17) # for fold-expression
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


message(STATUS "Building flag gems with C extensions")
include(FetchContent)
# dependencies: cuda toolkit
find_package(CUDAToolkit REQUIRED)
# dependencies: python
find_package(Python REQUIRED COMPONENTS Interpreter Development)
# torch
find_package(Torch MODULE REQUIRED) # This is the FindTorch.cmake
# dependencies: json
if(FLAGGEMS_USE_EXTERNAL_TRITON_JIT)
  find_package(TritonJIT 0.1.0 CONFIG REQUIRED)
else()
  set(TRITON_JIT_INSTALL ON) # install triton jit
  FetchContent_Declare(TritonJIT
    GIT_REPOSITORY https://github.com/FlagTree/libtriton_jit.git
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
