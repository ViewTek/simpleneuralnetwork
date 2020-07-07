cmake_minimum_required(VERSION 3.0.0)
project(neuralnetwork VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(neuralnetwork main.cpp)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-c++11-extensions")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
