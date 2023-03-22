cmake_minimum_required(VERSION 3.16)

file(GLOB LIB_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
add_library(stb STATIC ${LIB_FILES})