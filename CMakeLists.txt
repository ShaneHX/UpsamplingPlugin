# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(UpsamplePlugin LANGUAGES CXX CUDA)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Werror")

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# -------- BUILDING --------

# Add include directories
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE} ${CMAKE_SOURCE_DIR}/plugin/)

# Define clip plugin library target
add_library(upsampleplugin MODULE
  ${CMAKE_SOURCE_DIR}/plugin/UpsampleKernel.cu
  ${CMAKE_SOURCE_DIR}/plugin/UpsamplePlugin.cpp
  ${CMAKE_SOURCE_DIR}/plugin/UpsmapleKernel.h
  ${CMAKE_SOURCE_DIR}/plugin/UpsamplePlugin.h
)

# Use C++11
target_compile_features(upsampleplugin PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(upsampleplugin PRIVATE ${NVINFER_LIB})

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
set_target_properties(upsampleplugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
