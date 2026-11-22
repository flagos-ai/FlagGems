file(GLOB ATTENTION_SRCS
    *.cpp
    *.cu
)
add_subdirectory(attention)
add_library(flaggems_attention_ops OBJECT ${ATTENTION_SRCS})
target_link_libraries(flaggems_attention_ops PRIVATE flaggems_common)
target_include_directories(flaggems_attention_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
)
