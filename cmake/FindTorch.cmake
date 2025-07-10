# dependencies: torch
# use the current python interpreter's torch installation
if (NOT DEFINED Torch_ROOT)
  execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import torch;print(torch.utils.cmake_prefix_path)"
                  OUTPUT_VARIABLE Torch_ROOT
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  COMMAND_ECHO STDOUT
                  ECHO_OUTPUT_VARIABLE)
endif()
find_package(Torch CONFIG REQUIRED)

# message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
# get_target_property(TMP_TORCH_LINKED_LIBRARIES torch INTERFACE_LINK_LIBRARIES)
# message(STATUS "torch linked libraries: ${TMP_TORCH_LINKED_LIBRARIES}")


# message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
# get_target_property(TMP_TORCH_INTERFACE_INCLUDE_DIR torch INTERFACE_INCLUDE_DIRECTORIES)
# message(STATUS "torch linked libraries: ${TMP_TORCH_INTERFACE_INCLUDE_DIR}")
# depending on targets other than cmake variables has better composability
# since TritonJIT publicly depends on torch, It is better to transitive(recursively) resolve
# all the dependencies when other targets dependens on TritonJIT::triton_jit
# But cmake variable have Direcory Scope, thus, those variables "TORCH_LIBRARIES" does not propagate
# So we would depend on the target instead. Using an alias is for better naming convention to have
# targets inside some namespace
if (NOT TARGET Torch::Torch)
  add_library(Torch::Torch INTERFACE IMPORTED)
  target_include_directories(Torch::Torch INTERFACE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(Torch::Torch INTERFACE ${TORCH_LIBRARIES})
  target_compile_options(Torch::Torch INTERFACE ${TORCH_CXX_FLAGS})

  # add torch_python
  add_library(Torch::Torch_Python INTERFACE IMPORTED)
  find_library(torch_python_lib
    NAMES torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    REQUIRED)
  message(STATUS "find torch_python lib: ${torch_python_lib}")
  target_include_directories(Torch::Torch_Python INTERFACE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(Torch::Torch_Python INTERFACE ${torch_python_lib})
  target_compile_options(Torch::Torch_Python INTERFACE ${TORCH_CXX_FLAGS})
endif()

# Since multiple libraries or executables need to link each other, while the torch
# we use is a pre-built library, we need to ensure ABI consistency
add_compile_options(${TORCH_CXX_FLAGS})
message(STATUS "Using ABI for the whole project from torch: ${TORCH_CXX_FLAGS}")
