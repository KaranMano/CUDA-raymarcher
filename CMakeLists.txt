cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 60)
project(volume-renderer LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME} src/main.cu)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# Required for CUDA builds
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
