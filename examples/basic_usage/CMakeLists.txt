cmake_minimum_required(VERSION 3.17)
project(basic)
set(VS_STARTUP_PROJECT basic)
set(CMAKE_DEFAULT_STARTUP_PROJECT basic)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
set(GCC_COVERAGE_LINK_FLAGS "-lrt")

add_subdirectory(../../ ${CMAKE_CURRENT_BINARY_DIR}/tfexec)

add_executable(basic main.cpp)
target_include_directories(basic PRIVATE
    ../../include
    ../../lib/cppflow/include)
target_link_libraries(basic tfexec)
