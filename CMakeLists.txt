cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(FlagGems
  LANGUAGES CXX
  VERSION 0.1.0)

include(cmake/Options.cmake)
include(cmake/Settings.cmake)

if (BUILD_C_EXTENSIONS)
  message(STATUS "Building flag gems with C extensions")
  include(cmake/Dependencies.cmake)
  add_subdirectory(lib)
  if(BUILD_CTESTS)
    enable_testing()
    add_subdirectory(ctests)
  endif()

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

  pybind11_add_module(c_operators "${PROJECT_SOURCE_DIR}/src/flag_gems/csrc/cstub.cpp")
  target_link_libraries(c_operators PRIVATE operators)
  # Installing the extension module to the root of the package
  install(TARGETS c_operators
          LIBRARY DESTINATION ".")
  set_target_properties(
            c_operators PROPERTIES INSTALL_RPATH "${_rpath_portable_origin}/${CMAKE_INSTALL_LIBDIR}")
endif()
