find_program(FLAGTREE_EXECUTABLE
    NAMES flagtree tle
    PATHS ENV PATH
    DOC "FlagTree Triton compiler for Iluvatar"
)

if (FLAGTREE_EXECUTABLE)
    set(FlagTree_FOUND TRUE)
    message(STATUS "Found flagtree compiler: ${FLAGTREE_EXECUTABLE}")
else()
    message(FATAL_ERROR "flagtree compiler not found, please add to PATH")
endif()
