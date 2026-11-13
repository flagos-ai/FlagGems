find_path(ILUVATAR_INCLUDE_DIR hi_runtime.h
    PATHS /usr/local/iluvatar /opt/iluvatar
    PATH_SUFFIXES include
)
find_library(ILUVATAR_RUNTIME_LIB
    NAMES hiruntime
    PATHS /usr/local/iluvatar /opt/iluvatar
    PATH_SUFFIXES lib64 lib
)

if (ILUVATAR_INCLUDE_DIR AND ILUVATAR_RUNTIME_LIB)
    set(Iluvatar_FOUND TRUE)
    message(STATUS "Iluvatar SDK found: ${ILUVATAR_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Iluvatar SDK not installed")
endif()
