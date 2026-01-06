pybind11_add_module(c_operators cstub.cpp)
target_link_libraries(c_operators PRIVATE operators)
set_target_properties(c_operators PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/${CMAKE_INSTALL_LIBDIR}")

if (FLAGGEMS_INSTALL)
  # Installing the extension module to the root of the package
  install(TARGETS c_operators
          LIBRARY DESTINATION ".")
endif()
