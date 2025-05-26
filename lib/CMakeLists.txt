add_library(operators
            SHARED
            utils.cpp
            add.cpp
            sum.cpp
            mm.cpp
            rms_norm.cpp
            fused_add_rms_norm.cpp)
target_include_directories(operators
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(operators
  PUBLIC Torch::Torch
  PRIVATE TritonJIT::triton_jit)
add_library(FlagGems::operators ALIAS operators)


if (FLAGGEMS_INSTALL)
  include(GNUInstallDirs)
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/flag_gems"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS operators
    EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems
    NAMESPACE FlagGems::
    FILE FlagGemsTargets.cmake)
  export(EXPORT FlagGemsTargets FILE "${PROJECT_BINARY_DIR}/FlagGemsTargets.cmake")
endif()
