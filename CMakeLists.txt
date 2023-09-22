cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 60)
project(volume-renderer LANGUAGES CXX CUDA)

find_package(CUDAToolkit)
find_package( OpenGL REQUIRED )

set( GLFW_BUILD_DOCS OFF CACHE BOOL  "GLFW lib only" )
set( GLFW_INSTALL OFF CACHE BOOL  "GLFW lib only" )
add_definitions( -DGLEW_STATIC )

add_subdirectory( external/glfw )
add_subdirectory( external/glew )
add_subdirectory( external/stb )
add_subdirectory( external/imgui )

if( MSVC )
    SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup" )
endif()

file( GLOB_RECURSE SOURCE_FILES src/* )
list(REMOVE_ITEM SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/benchmark.cu)
add_executable( ${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries( ${CMAKE_PROJECT_NAME}
	PRIVATE glfw
	PRIVATE ${OPENGL_LIBRARIES}
	PRIVATE libglew_static
	PRIVATE imgui
	PRIVATE stb
)
target_include_directories( ${CMAKE_PROJECT_NAME}
	PRIVATE ${OPENGL_INCLUDE_DIR}
	PRIVATE external/imgui
	PRIVATE external/stb
)
	
set_property( TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17 )
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

file( GLOB_RECURSE SOURCE_FILES src/* )
list(REMOVE_ITEM SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/main.cu)
add_executable( benchmark ${SOURCE_FILES})
target_link_libraries( benchmark
	PRIVATE stb
)
target_include_directories( benchmark
	PRIVATE external/stb
)
	
set_property( TARGET benchmark PROPERTY CXX_STANDARD 17 )
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)