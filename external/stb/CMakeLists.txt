cmake_minimum_required(VERSION 3.16)

file(GLOB_RECURSE LIB_FILES *)
add_library(stb STATIC ${LIB_FILES})