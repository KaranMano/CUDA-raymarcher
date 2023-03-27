cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 60)
project(volume-renderer LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

add_subdirectory( external/stb )

file( GLOB_RECURSE SOURCE_FILES src/* )
add_executable( ${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries( ${CMAKE_PROJECT_NAME}
	PRIVATE stb
)
target_include_directories( ${CMAKE_PROJECT_NAME}
	PRIVATE external/stb
	PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)
	
set_property( TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)