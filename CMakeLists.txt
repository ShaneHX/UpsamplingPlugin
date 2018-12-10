cmake_minimum_required(VERSION 3.0.0)
project(plugin_test VERSION 0.1.0)


FIND_PACKAGE(OpenCV)

FIND_PACKAGE(CUDA)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

include(CTest)
enable_testing()

add_compile_options(-std=c++11)

include_directories(${CMAKE_SOURCE_DIR}/plugin)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/plugin/*.cpp ${CMAKE_SOURCE_DIR}/plugin/*.cu)

#add_executable(plugin_test main.cpp ${SOURCE_FILES})
set(CUDA_NVCC_FLAGS -std=c++11)
cuda_add_executable(plugin_test main.cpp ${SOURCE_FILES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

TARGET_LINK_LIBRARIES(plugin_test ${OpenCV_LIBS} nvcaffe_parser nvinfer cudnn)