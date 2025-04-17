include(FetchContent)

# dependencies: cuda toolkit
find_package(CUDAToolkit REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(Torch MODULE REQUIRED) # This is the FindTorch.cmake

# dependencies: json
if (FLAGGEMS_USE_EXTERNAL_TRITON_JIT)
  find_package(TritonJIT VERSION 0.1.0 CONFIG REQUIRED)
else()
    FetchContent_Declare(TritonJIT
      GIT_REPOSITORY https://github.com/iclementine/libtorch_example.git
      # SOURCE_DIR /home/clement/projects/libtorch_example # use local source dir in development
    )
    FetchContent_MakeAvailable(TritonJIT)
endif()

if (FLAGGEMS_USE_EXTERNAL_PYBIND11)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ECHO STDOUT
    ECHO_OUTPUT_VARIABLE)
  find_package(pybind11 CONFIG REQUIRED)
else()
  FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11)
  FetchContent_MakeAvailable(pybind11)
endif()
