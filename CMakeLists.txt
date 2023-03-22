cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 60)
project(volume-renderer LANGUAGES CXX CUDA)

add_subdirectory( external/stb )
message(STB_LIB_HEADERS="${STB_LIB_HEADERS}")

set( SOURCE_FILES
	src/main.cu
)
add_executable( ${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries( ${CMAKE_PROJECT_NAME}
	PRIVATE stb
)
target_include_directories( ${CMAKE_PROJECT_NAME}
	PRIVATE external/stb
)
	
set_property( TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
