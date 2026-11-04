cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(FlagGems LANGUAGES CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ------------------------------- Global Project Settings -------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(FLAGGEMS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${FLAGGEMS_ROOT}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${FLAGGEMS_ROOT}/lib)

# ------------------------------- RPATH Settings -------------------------------
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_rpath_portable_origin "@loader_path")
else()
  set(_rpath_portable_origin $ORIGIN)
endif()
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ------------------------------- Project Options -------------------------------
option(FLAGGEMS_USE_EXTERNAL_TRITON_JIT "whether to use external triton jit library" OFF)
option(FLAGGEMS_USE_EXTERNAL_PYBIND11 "whether to use external pybind11 library" ON)
option(FLAGGEMS_BUILD_C_EXTENSIONS "whether to build c extension" OFF)
option(FLAGGEMS_BUILD_CTESTS "Whether to build CPP unit tests" ${FLAGGEMS_BUILD_C_EXTENSIONS})
option(FLAGGEMS_INSTALL "Whether to install the package" ${PROJECT_IS_TOP_LEVEL})
# Iluvatar/FlagTree 配套选项
option(FLAGGEMS_BUILD_TESTS "Build C++ tests (Iluvatar path)" OFF)
option(FLAGGEMS_BUILD_BINDING "Build pybind11 binding (Iluvatar path)" OFF)

# ------------------------------- Backend Selection -------------------------------
# set(FLAGGEMS_BACKEND "IX" CACHE STRING "Target backend: CUDA, IX, MUSA, NPU" FORCE)
set(FLAGGEMS_BACKEND "IX" CACHE STRING "Target backend: CUDA, IX, MUSA, NPU" )
set_property(CACHE FLAGGEMS_BACKEND PROPERTY STRINGS "CUDA" "IX" "MUSA" "NPU")
if(NOT FLAGGEMS_BACKEND MATCHES "^(CUDA|IX|MUSA|NPU)$")
    message(FATAL_ERROR "Invalid FLAGGEMS_BACKEND: ${FLAGGEMS_BACKEND}")
endif()
message(STATUS "Building FlagGems with backend: ${FLAGGEMS_BACKEND}")

# 仅CUDA后端启用CUDA语言，IX不启用
if(FLAGGEMS_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
endif()

# Backend compile definitions
if(FLAGGEMS_BACKEND STREQUAL "CUDA")
    add_compile_definitions(FLAGGEMS_USE_CUDA)
elseif(FLAGGEMS_BACKEND STREQUAL "IX")
    add_compile_definitions(FLAGGEMS_USE_IX)
elseif(FLAGGEMS_BACKEND STREQUAL "NPU")
    add_compile_definitions(FLAGGEMS_USE_NPU)
elseif(FLAGGEMS_BACKEND STREQUAL "MUSA")
    add_compile_definitions(FLAGGEMS_USE_MUSA)
endif()

# ------------------------------- Dependency & Source Branch -------------------------------
if(FLAGGEMS_BACKEND STREQUAL "IX")
    message(STATUS "Iluvatar(IX) backend: Skip C++ extension build, use FlagTree JIT only")
    # 【关键注释】不查找Iluvatar SDK、FlagTree，不编译cpp_ext
    # find_package(Iluvatar REQUIRED)
    # find_package(FlagTree REQUIRED)
    # include_directories(${FLAGGEMS_ROOT}/include)
    # include_directories(${ILUVATAR_INCLUDE_DIR})
    # add_subdirectory(src/flag_gems/cpp_ext)
    # if (FLAGGEMS_BUILD_TESTS)
    #     enable_testing()
    #     add_subdirectory(ctests)
    # endif()
elseif(FLAGGEMS_BUILD_C_EXTENSIONS)
    # CUDA / MUSA / NPU 原始多后端 C extension 路径
    message(STATUS "Building flag gems with C extensions")
    include(FetchContent)

    find_package(Python REQUIRED COMPONENTS Interpreter Development)

    if(FLAGGEMS_BACKEND STREQUAL "CUDA")
        find_package(CUDAToolkit REQUIRED)
    elseif(FLAGGEMS_BACKEND STREQUAL "NPU")
        include(BackendNPU)
    elseif(FLAGGEMS_BACKEND STREQUAL "MUSA")
        include(BackendMUSA)
    endif()

    find_package(Torch MODULE REQUIRED)
# TritonJIT
    if(FLAGGEMS_USE_EXTERNAL_TRITON_JIT)
      find_package(TritonJIT 0.1.0 CONFIG REQUIRED)
    else()
      set(TRITON_JIT_INSTALL ON)
      set(BACKEND ${FLAGGEMS_BACKEND} CACHE STRING "Backend for TritonJIT")
      FetchContent_Declare(TritonJIT
        GIT_REPOSITORY https://github.com/flagos-ai/libtriton_jit
        GIT_TAG master
      )
      FetchContent_MakeAvailable(TritonJIT)
    endif()

    # pybind11
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

    # GTest
    if(FLAGGEMS_BUILD_CTESTS)
      set(INSTALL_GTEST OFF)
      FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0
      )
      FetchContent_MakeAvailable(googletest)
    endif()

    # Triton version check
    execute_process(
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

    add_subdirectory(lib)
    add_subdirectory(src/flag_gems/csrc)
    if(FLAGGEMS_BUILD_CTESTS)
      enable_testing()
      add_subdirectory(ctests)
    endif()
 # Install
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
      install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTorch.cmake
              DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems)
    endif()
endif()
