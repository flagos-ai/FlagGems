pybind11_add_module(c_operators cstub.cpp)
target_link_libraries(c_operators PUBLIC Torch::Torch_Python PRIVATE operators)
set_target_properties(c_operators PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/${CMAKE_INSTALL_LIBDIR}")


pybind11_add_module(aten_patch aten_patch.cpp)
target_link_libraries(aten_patch PUBLIC Torch::Torch_Python PRIVATE operators)
set_target_properties(aten_patch PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/${CMAKE_INSTALL_LIBDIR}")

if (FLAGGEMS_INSTALL)
  # Installing the extension module to the root of the package
  install(TARGETS c_operators aten_patch
          LIBRARY DESTINATION ".")
endif()
