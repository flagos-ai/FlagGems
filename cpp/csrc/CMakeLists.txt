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

pybind11_add_module(c_operators cstub.cpp)
target_link_libraries(c_operators PUBLIC Torch::Torch_Python PRIVATE operators)
set_target_properties(c_operators PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/${CMAKE_INSTALL_LIBDIR}"
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON)
target_compile_options(c_operators PRIVATE -std=c++20)


pybind11_add_module(aten_patch aten_patch.cpp)
target_link_libraries(aten_patch PUBLIC Torch::Torch_Python PRIVATE operators)
set_target_properties(aten_patch PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/${CMAKE_INSTALL_LIBDIR}"
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON)
target_compile_options(aten_patch PRIVATE -std=c++20)

# Propagate pointwise_dynamic feature flag to pybind11 modules so that
# #ifdef FLAGGEMS_POINTWISE_DYNAMIC guards in cstub.cpp / aten_patch.cpp
# are consistent with the operators library.
if(FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP)
  target_compile_definitions(c_operators PRIVATE FLAGGEMS_POINTWISE_DYNAMIC)
  target_compile_definitions(aten_patch  PRIVATE FLAGGEMS_POINTWISE_DYNAMIC)
endif()

if (FLAGGEMS_INSTALL)
  # Installing the extension module to the root of the package
  install(TARGETS c_operators aten_patch
          LIBRARY DESTINATION ".")
endif()
