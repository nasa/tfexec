cmake_minimum_required(VERSION 3.17)
include(ExternalProject)
include(FetchContent)

# Fetch the external library for Tensorflow C-Language Bindings
if (WIN32)
  set(tensorflowurl "http://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-2.3.0.zip")
else()
  set(tensorflowurl "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.3.0.tar.gz")
endif()

FetchContent_Declare(tensorflowcontent
  URL ${tensorflowurl})
FetchContent_GetProperties(tensorflowcontent)
if (NOT tensorflowcontent_POPULATED)
  FetchContent_MakeAvailable(tensorflowcontent)
endif()

add_library(Tensorflow SHARED IMPORTED)
if (WIN32)
  set_property(TARGET Tensorflow PROPERTY
    IMPORTED_LOCATION "${tensorflowcontent_SOURCE_DIR}/lib/tensorflow.dll")
  set_property(TARGET Tensorflow PROPERTY
    IMPORTED_IMPLIB "${tensorflowcontent_SOURCE_DIR}/lib/tensorflow.lib")
else()
  set_property(TARGET Tensorflow PROPERTY
    IMPORTED_LOCATION "${tensorflowcontent_SOURCE_DIR}/lib/libtensorflow.so")
endif()
target_include_directories(Tensorflow INTERFACE "${tensorflowcontent_SOURCE_DIR}/include")

# CPPFlow target
add_library(CPPFlow INTERFACE)
set(CMAKE_CXX_STANDARD 14)
target_include_directories(CPPFlow INTERFACE "lib/cppflow/include")

# Specify the TFExec library
project(tfexec)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")

set(GCC_COVERAGE_LINK_FLAGS "-lrt")

add_library(tfexec STATIC src/tfexec/tfexec.cpp)

set_property(TARGET tfexec PROPERTY CXX_STANDARD 17)
set_property(TARGET tfexec PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(tfexec PRIVATE ./include)
target_link_libraries(tfexec Tensorflow CPPFlow)
