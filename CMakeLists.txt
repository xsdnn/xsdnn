cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(xsDNN VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS      "Set to ON to build tests"              ON)

if(BUILD_TESTS)
    add_subdirectory(test)
endif(BUILD_TESTS)
